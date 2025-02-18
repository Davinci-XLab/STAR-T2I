import sys
sys.path.insert(0, '../../')
from dataset.test_fid_coco import batched_iterator, transform_image, batched_iterator_MJHQ
from models import VAR, VQVAE, build_vae_var
from models.text_encoder import build_text
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils.utils import format_sentence
from torch.utils.data import DataLoader, Dataset
import random
from torchvision import transforms
import torch
import datasets as hf_datasets
from PIL import Image
import os.path as osp
import pdb
import numpy as np
import os
from metrics.compare_models.eval_fid import transform_image_fid


import argparse
import json
import os
import torch
import numpy as np
from PIL import Image
from tqdm import trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything

from models import VAR, VQVAE, build_vae_var
from models.text_encoder import build_text
# from model_setup import build_vae_var, build_text  # 假设这些是来自prepare_images中的自定义模块

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_file",
        type=str,
        default='/home/disk2/nfs/maxiaoxiao/benchmarks/GenEval/prompts/evaluation_metadata.jsonl',
        help="JSONL file containing metadata with prompts for each generation"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/nfs/nfs-141/maxiaoxiao/eval_results/GenEval_1024_2",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--gen_reso",
        type=int,
        default=1024,
        help="Resolution of the generated images"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=30,
        help="Depth setting for the model"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of samples to generate per prompt"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility"
    )
    return parser.parse_args()

def main(opt):
    # 加载 metadata
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    # 准备模型
    savedir_pred = os.path.join(opt.outdir, 'prediction')
    savedir_gt = os.path.join(opt.outdir, 'reference')
    os.makedirs(savedir_pred, exist_ok=True)
    os.makedirs(savedir_gt, exist_ok=True)

    depth=30
    patch_nums =[1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64]
    # var_ckpt='/home/nfs/nfs-142/maxiaoxiao/workspace/var_rope_d30_1024_vqa/ar-ckpt-last.pth'
    var_ckpt='/home/nfs/nfs-142/maxiaoxiao/workspace/var_rope_d30_1024_vqa_tune/ar-ckpt-ep0-iter3000.pth'
    enable_logit_norm=True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae_ckpt='/home/disk2/nfs/maxiaoxiao/ckpts/FoundationVision-var/vae_ch160v4096z32.pth'
    texenc_ckpt='/home/disk2/nfs/maxiaoxiao/ckpts/stable-diffusion-2-1'

    vae_local, var_wo_ddp = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        depth=depth, shared_aln=False, attn_l2_norm=True,
        enable_cross=True,
        in_dim_cross=1024,#TODO:换成从text enc得到的参数 
        flash_if_available=False, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=5e-5, init_head=0.02, init_std=-1,
        rope_emb=True,lvl_emb=True,
        enable_logit_norm=enable_logit_norm,
        enable_adaptive_norm=False,
        train_mode='none',
        rope_theta=10000,
        rope_norm=64.0
    )

    var_wo_ddp.load_state_dict(torch.load(var_ckpt, map_location='cpu')["trainer"]["var_wo_ddp"], strict=True)
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

    var_wo_ddp.eval()
    vae_local.eval()

    # 提取文本特征
    text_encoder = build_text(pretrained_path=texenc_ckpt, device=device)
    text_encoder.eval()

    for index, metadata in enumerate(metadatas):
        base_seed = opt.seed

        outpath = os.path.join(savedir_pred, f"{index:05}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata['prompt']
        print(f"Prompt ({index + 1}/{len(metadatas)}): {prompt}")
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        # 逐次生成两张图片
        all_samples = []
        for batch_index in range(2):  # 每次生成两张，共生成两次
            current_seed = base_seed + batch_index  # 使用不同的随机种子
            seed_everything(current_seed)

            with torch.no_grad():
                # 提取文本特征
                prompt_embeds, prompt_attention_mask, pooled_embed = \
                    text_encoder.extract_text_features([prompt] * 2 + [""] * 2)
                with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                    recon_B3HW = var_wo_ddp.autoregressive_infer_cfg(
                        B=2,
                        label_B=None,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        encoder_pool_feat=pooled_embed,
                        cfg=4.0, top_k=600, top_p=0.8, g_seed=current_seed,
                        more_smooth=False, w_mask=False
                    )

                for i in range(2):
                    img_pred = (recon_B3HW[i].permute(1, 2, 0).cpu().numpy() * 255.).astype(np.uint8)
                    img = Image.fromarray(img_pred)
                    img.save(os.path.join(outpath, f"{batch_index * 2 + i:05}.png"))

                    # 保存生成的图片
                    all_samples.append(ToTensor()(img))

        # 保存图片网格
        if opt.n_samples > 1:
            grid = make_grid(all_samples, nrow=opt.n_samples)
            grid = (255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()).astype(np.uint8)
            grid_img = Image.fromarray(grid)
            grid_img.save(os.path.join(outpath, 'grid.png'))

    print("Generation completed.")

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
