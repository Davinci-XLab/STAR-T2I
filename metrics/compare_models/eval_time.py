import sys
sys.path.insert(0,'../../')

from diffusers import StableDiffusionPipeline,DiffusionPipeline,PixArtAlphaPipeline
from models import build_vae_var
from models.text_encoder import build_text
from diffusers import StableDiffusionXLPipeline,EulerAncestralDiscreteScheduler
from PIL import Image
import torch

import time

if __name__=="__main__":
    prompt=['Australian rainbow serpent festival during the night time closeup photo of cyberpunk teen girl, photorealistic portrait lens 50mm professional lens.']*5
    device="cuda"

    # # sdv2.1
    # repo_id = "/home/disk2/mxx/ckpts/stable-diffusion-2-1"
    # pipe = StableDiffusionPipeline.from_pretrained(repo_id, use_safetensors=True).to(device)
    # start_time=time.time()
    # image_B3HW=pipe(prompt,height=512,width=512).images
    # print('time for sdv2.1 is ',time.time()-start_time)

    # # sdxl
    # base = StableDiffusionXLPipeline.from_pretrained(
    #     "/home/disk2/mxx/ckpts/SDXL-512",
    #     use_safetensors=True
    # ).to(device)
    # base.scheduler = EulerAncestralDiscreteScheduler.from_config(base.scheduler.config)
    # start_time=time.time()
    # oup_B3HW = base(
    #     prompt=prompt,
    #     num_inference_steps=40,
    #     height=512,width=512,
    #     target_size=(1024, 1024),
    #     original_size=(4096, 4096)
    # ).images
    # print('time for sdxl is ',time.time()-start_time)
    # oup_B3HW[0].save('./sdxl_example.png')


    # # playground2.5
    # repo_id = "/home/disk2/mxx/ckpts/playground-v2.5-1024px-aesthetic"
    # # repo_id = "/home/disk2/mxx/ckpts/playground-v2-512px-base"
    # pipe = DiffusionPipeline.from_pretrained(
    #     repo_id, use_safetensors=True,
    #         add_watermarker=False
    # ).to(device)
    # start_time=time.time()
    # image_B3HW=pipe(prompt,height=512,width=512).images
    # print('time for playground2.5 is ',time.time()-start_time)
    # image_B3HW[0].save('./playground2-5_example.png')

    # pixart
    repo_id = "/home/disk1/liangtao/comm/PixArt-alpha--PixArt-XL-2-512x512"
    pipe = PixArtAlphaPipeline.from_pretrained(repo_id, use_safetensors=True).to(device)
    start_time=time.time()
    image_B3HW=pipe(prompt,height=512,width=512).images
    print('time for pixart is ',time.time()-start_time)
    for i in range(len(prompt)):
        image_B3HW[i].save('pixart_example_%d.png'%i)

    '''
    # ours
    texenc_ckpt='/home/disk2/mxx/ckpts/stable-diffusion-2-1'
    vae_ckpt='/home/disk2/mxx/ckpts/FoundationVision-var/vae_ch160v4096z32.pth'
    var_ckpt='/home/disk2/mxx/workspace/var_rope_d30_512/ar-ckpt-ep0-iter125000.pth'
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16,24,32)
    vae_local, var_wo_ddp = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        depth=30, shared_aln=False, attn_l2_norm=False,
        in_dim_cross=1024,#TODO:换成从text enc得到的参数
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=5e-5, init_head=0.02, init_std=-1,
        rope_emb=True,lvl_emb=True,
    )
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    vae_local.to(device)
    text_encoder=build_text(pretrained_path=texenc_ckpt,device=device)
    vae_local.eval();var_wo_ddp.eval();text_encoder.eval()

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    
    with torch.no_grad():
        with torch.inference_mode():
            start_time=time.time()
            prompt_embeds,prompt_attention_mask,pooled_embed = text_encoder.extract_text_features(prompt+[""])
            with torch.autocast('cuda', cache_enabled=True,dtype=torch.float32): 
                recon_B3HW = var_wo_ddp.autoregressive_infer_cfg(B=1, label_B=None, 
                                                                encoder_hidden_states=prompt_embeds,
                                                                encoder_attention_mask=prompt_attention_mask,
                                                                encoder_pool_feat=pooled_embed,
                                                                cfg=4, top_k=900, 
                                                                top_p=0.95,g_seed=42)#smooth会导致细节减少，但可靠性提升
    print('time for ours is ',time.time()-start_time)
    '''

    