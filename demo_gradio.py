import random
import time
import torch 
import gradio as gr 
import numpy as np
from PIL import Image

from models import build_vae_var
from models.text_encoder import build_text

torch.set_grad_enabled(False)

# 加载模型到 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

B_ = 1
# 模型配置
V = 4096
ch = 160
Cvae = 32
share_quant_resi = 4

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

depth=30

enable_logit_norm=True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

vae_ckpt='/path/to/vae'
texenc_ckpt='/path/to/SDXL_CLIP'

patch_nums = [1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64]
var_ckpt='./ckpts/var_rope_d30_1024_sampler_mask_0108-ar-ckpt-ep2-iter6000.pth'

# patch_nums = [1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64]
# var_ckpt='./ckpts/var_rope_d30_1024_sampler_mask-ar-ckpt-ep1-iter26000.pth'

# 构建 VAE 和 VAR 模型
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
    enable_adaptive_norm=True,
    train_mode='none',
    rope_theta=10000,
    rope_norm=64.0,
    sample_from_idx=9
)

# 加载模型状态
var_wo_ddp.load_state_dict(torch.load(var_ckpt, map_location='cpu')["trainer"]["var_wo_ddp"], strict=True)
vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

var_wo_ddp.eval()
vae_local.eval()

# 构建文本编码器
text_encoder, _ = build_text(pretrained_path=texenc_ckpt,device=device)
text_encoder.eval()

def run_model(prompt, more_smooth, cfg, top_k, top_p, seed):
    # 确保随机种子大于等于0
    if seed < 0:
        raise ValueError("Seed must be greater than or equal to 0")
    
    # 设置随机种子
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # 记录开始时间
    start_time = time.time()

    with torch.no_grad():
        with torch.inference_mode():
            prompt_embeds,prompt_attention_mask,pooled_embed = text_encoder.extract_text_features([prompt]+[""]*B_)
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                recon_B3HW = var_wo_ddp.autoregressive_infer_cfg(B=B_, label_B=None, 
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_attention_mask,
                            encoder_pool_feat=pooled_embed,
                            cfg=cfg, top_k=top_k,
                            top_p=top_p, g_seed=seed,
                            more_smooth=False,
                            w_mask=True,
                            sample_version='1024'
                        )
            
            images = []
            for i in range(B_):
                img_pred = (recon_B3HW[i].permute(1, 2, 0).cpu().numpy() * 255.).astype(np.uint8)
                img = Image.fromarray(img_pred)
                images.append(img)
    
    # 记录结束时间
    end_time = time.time()
    inference_time = end_time - start_time
    
    return images, f"Inference Time: {inference_time:.2f} seconds"

# 创建 Gradio 界面
interface = gr.Interface(
    fn=run_model,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your text prompt here...", label="Text Prompt"),
        gr.Checkbox(label="More Smooth"),
        gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=4.0, label="CFG"),
        gr.Slider(minimum=1, maximum=4096, step=1, value=600, label="Top K"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.8, label="Top P"),
        gr.Number(label="Seed", value=42, minimum=0)  # 设置 Seed 输入的最小值为0
    ],
    outputs=[
        gr.Gallery(label="Generated Images"),
        gr.Textbox(label="Inference Time")
    ],
    title="STAR Text-to-Image Generation",
    description="Generate images from text prompts using a STAR model."
)

if __name__ == "__main__":
    interface.launch(server_name='0.0.0.0', server_port=8158, max_threads=1, share=False)
