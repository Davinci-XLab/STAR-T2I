import argparse
import copy
import datetime
import os
import random
import time

import numpy as np
import torch
import torchvision
from PIL import Image

from models import build_vae_var
from models.text_encoder import build_text


def save_images(sample_imgs, sample_folder_dir, store_separately, prompts, seed=1):
    if not store_separately and len(sample_imgs) > 1:
        grid = torchvision.utils.make_grid(sample_imgs, nrow=12)
        grid_np = grid.to(torch.float16).permute(1, 2, 0).mul_(255).cpu().numpy()

        os.makedirs(sample_folder_dir, exist_ok=True)
        grid_np = Image.fromarray(grid_np.astype(np.uint8))
        grid_np.save(os.path.join(sample_folder_dir, f"sample_images_{seed}.png"))
        print(f"Example images are saved to {sample_folder_dir}")
    else:
        # bs, 3, r, r
        sample_imgs_np = sample_imgs.mul_(255).cpu().numpy()
        num_imgs = sample_imgs_np.shape[0]
        os.makedirs(sample_folder_dir, exist_ok=True)
        for img_idx in range(num_imgs):
            cur_img = sample_imgs_np[img_idx]
            cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
            cur_img_store = Image.fromarray(cur_img)
            cur_img_store.save(os.path.join(sample_folder_dir, f"{img_idx:06d}.png"))
            print(f"Image {img_idx} saved.")

    with open(os.path.join(sample_folder_dir, "prompt.txt"), "w") as f:
        f.write("\n".join(prompts))


def main(args):
    device = torch.device("cuda")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    vae_model, var_model = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=device, patch_nums=args.patch_nums,
        depth=args.depth, shared_aln=False, attn_l2_norm=True,
        enable_cross=True,
        in_dim_cross=1024,
        flash_if_available=False, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=5e-5, init_head=0.02, init_std=-1,
        rope_emb=True,lvl_emb=True,
        enable_logit_norm=args.enable_logit_norm,
        enable_adaptive_norm=False,
        train_mode='none',
        rope_theta=10000,
        rope_norm=64.0,
        sample_from_idx=9
    )

    vae_model.load_state_dict(torch.load(args.vae_path, map_location='cpu'), strict=True)
    var_model.load_state_dict(torch.load(args.model_path, map_location='cpu')["trainer"]["var_wo_ddp"], strict=True)

    vae_model.eval()
    var_model.eval()

    text_encoder, _ = build_text(pretrained_path=args.text_model_path,device=device)
    text_encoder.eval()

    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_list:
        prompts = args.prompts
    else:
        prompts = [
            "A red car.",
            "A cute dog.",
            "A banana in a plate.",
            "A giraffe.",
            "A red car and a white sheep.",
            "A blue bird on a tree.",
            "A green apple on the table.",
            "A green cup and a blue cell phone.",
        ]

    start_time = time.time()

    batch_size = args.batch_size
    for batch_prompt in range(0, len(prompts), batch_size):
        prompt = prompts[batch_prompt:batch_prompt+batch_size]
        with torch.no_grad():
            with torch.inference_mode():
                prompt_embeds,prompt_attention_mask,pooled_embed = text_encoder.extract_text_features(prompt+[""]*batch_size)
                with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                    recon_B3HW = var_model.autoregressive_infer_cfg(B=batch_size, label_B=None, 
                                encoder_hidden_states=prompt_embeds,
                                encoder_attention_mask=prompt_attention_mask,
                                encoder_pool_feat=pooled_embed,
                                cfg=args.cfg, top_k=args.top_k,
                                top_p=args.top_p, g_seed=args.seed,
                                more_smooth=False,
                                w_mask=True,
                                sample_version='1024'
                            )
        
        output_imgs = recon_B3HW if not "output_imgs" in locals() else torch.cat([output_imgs, recon_B3HW], dim=0)
    
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Generate {len(prompts)} images take {inference_time:2f}s.")

    save_images(
        output_imgs.clone(), args.sample_folder_dir, args.store_seperately, prompts, args.seed
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to STAR model.",
        default="pretrained_models/STAR.pth",
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model.",
        default="pretrained_models/SDXl_CLIP",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        help="The path to VAE model.",
        default="pretrained_models/VAE.pth",
    )
    parser.add_argument(
        "--cfg", type=float, help="Classifier-free guidance scale.", default=4.5
    )
    parser.add_argument(
        "--top_k",
        type=int,
        help="Top-k sampling parameter for generation.",
        default=600,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="Top-p sampling parameter for generation.",
        default=0.8,
    )
    parser.add_argument(
        "--patch_nums",
        type=list,
        help="The patch numbers for the model.",
        default=[1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64],
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="The depth of the model.",
        default=30,
    )
    parser.add_argument(
        "--enable_logit_norm",
        type=bool,
        help="Enable logit normalization.",
        default=True,
    )
    parser.add_argument(
        "--more_smooth",
        type=bool,
        help="Turn on for more visually smooth samples.",
        default=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The batch size for generation.",
        default=1,
    )
    parser.add_argument(
        "--sample_folder_dir",
        type=str,
        help="The folder where the image samples are stored",
        default="samples/",
    )
    parser.add_argument(
        "--store_seperately",
        help="Store image samples in a grid or separately, set to False by default.",
        action="store_true",
    )
    parser.add_argument("--prompt", type=str, help="A single prompt.", default="")
    parser.add_argument("--prompt_list", type=list, default=[])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)

    args = parser.parse_args()

    main(args)
