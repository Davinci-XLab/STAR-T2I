import sys
sys.path.insert(0, '../../')
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models import VAR, VQVAE, build_vae_var
from models.text_encoder import build_text

# Custom image generation function
def generate_images_for_prompt(prompt, model, text_encoder, num_images=4, seed=None):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    images = []
    
    with torch.no_grad():
        with torch.inference_mode():
            prompt_embeds, prompt_attention_mask, pooled_embed = text_encoder.extract_text_features([prompt]*num_images +[""]*num_images)
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                recon_image = model.autoregressive_infer_cfg(
                    B=num_images,
                    label_B=None,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    encoder_pool_feat=pooled_embed,
                    cfg=4.0, top_k=600, top_p=0.8, g_seed=seed,
                    more_smooth=False, w_mask=False
                )
            for i in range(num_images):
                img_pred = (recon_image[i].permute(1, 2, 0).cpu() * 255).numpy().astype(np.uint8)
                images.append(Image.fromarray(img_pred))
    return images

# Function to save the grid image
def save_image_grid(images, save_path, grid_size=(2, 2), img_size=(1024, 1024)):
    grid_img = Image.new('RGB', (grid_size[1] * img_size[1], grid_size[0] * img_size[0]))
    for idx, img in enumerate(images):
        x = idx % grid_size[1] * img_size[1]
        y = idx // grid_size[1] * img_size[0]
        grid_img.paste(img, (x, y))
    grid_img.save(save_path)


def build_model():
    var_ckpt='/home/nfs/nfs-40/maxiaoxiao/workspace/var_rope_d30_fsdp_1024_norm/ar-ckpt-ep0-iter82000.pth'
    texenc_ckpt='/home/disk2/nfs/maxiaoxiao/ckpts/stable-diffusion-2-1'
    vae_ckpt='/home/disk2/nfs/maxiaoxiao/ckpts/FoundationVision-var/vae_ch160v4096z32.pth'
    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth=30
    patch_nums =[1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64]
    var_ckpt='/home/nfs/nfs-142/maxiaoxiao/workspace/var_rope_d30_1024/ar-ckpt-ep0-iter48000.pth'
    vae_local, var_wo_ddp = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        depth=depth, shared_aln=False, attn_l2_norm=True,
        enable_cross=True,
        in_dim_cross=1024,#TODO:换成从text enc得到的参数 
        flash_if_available=False, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=5e-5, init_head=0.02, init_std=-1,
        rope_emb=True,lvl_emb=True,
        enable_logit_norm=True,
        enable_adaptive_norm=False,
        train_mode='none',
        rope_theta=10000,
        rope_norm=64.0
    )
    var_wo_ddp.load_state_dict(torch.load(var_ckpt, map_location='cpu')["trainer"]["var_wo_ddp"], strict=True)
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    text_encoder=build_text(pretrained_path=texenc_ckpt,device=device)
    vae_local.eval();var_wo_ddp.eval();text_encoder.eval()
    return vae_local, var_wo_ddp, text_encoder



# Read prompts and generate images
prompts_dir = '/home/disk2/nfs/maxiaoxiao/benchmarks/DPGBench/dpg_bench/prompts'
output_dir = '/home/nfs/nfs-141/maxiaoxiao/eval_results/DPGBench_1024_2'
if not os.path.exists(output_dir):os.makedirs(output_dir)
_,model,text_encoder = build_model()

prompt_files = sorted([f for f in os.listdir(prompts_dir) if f.endswith('.txt')])

for idx, prompt_file in enumerate(prompt_files):
    with open(os.path.join(prompts_dir, prompt_file), 'r') as f:
        prompt = f.readline().strip()
    
    images = generate_images_for_prompt(prompt, model, text_encoder, num_images=4, seed=42)
    
    output_path = os.path.join(output_dir, f"{prompt_file.split('.')[0]}.png")
    save_image_grid(images, output_path)
    print(f"Saved: {output_path}")
    print(prompt)
