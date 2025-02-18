import sys
sys.path.insert(0,'../../../')
print(sys.path)

# from dataset.test_fid_coco import batched_iterator,transform_image,batched_iterator_MJHQ
from models import VAR, VQVAE, build_vae_var
from models.text_encoder import build_text
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils.utils import format_sentence
from torch.utils.data import DataLoader, Dataset
import random
from torchvision import transforms
import torch
from PIL import Image
import os.path as osp
import pdb
import numpy as np
import os
import json




def prepare_images(savedir_pred,sample_per_batch=10):
    # 一些参数
    select_sz=1#16

    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16,24,32)
    texenc_ckpt='/home/disk2/mxx/ckpts/stable-diffusion-2-1'
    vae_ckpt='/home/disk2/mxx/ckpts/FoundationVision-var/vae_ch160v4096z32.pth'
    var_ckpt='/home/disk2/mxx/workspace/var_rope_d30_512_quality_tuning/ar-ckpt-ep4-iter0.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae_local, var_wo_ddp = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        depth=30, shared_aln=False, attn_l2_norm=False,
        enable_cross=True,in_dim_cross=1024,#TODO:换成从text enc得到的参数
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=5e-5, init_head=0.02, init_std=-1,
        rope_emb=True,lvl_emb=True,
    )
    
    var_wo_ddp.load_state_dict(torch.load(var_ckpt, map_location='cpu')["trainer"]["var_wo_ddp"], strict=True)
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    text_encoder=build_text(pretrained_path=texenc_ckpt,device=device)


    prompt_with_id_list = []
    with open('/home/disk2/mxx/VAR/metrics/image_reward/benchmark/benchmark-prompts.json', "r") as f:
        prompt_with_id_list = json.load(f)
    num_prompts = len(prompt_with_id_list)

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')


    vae_local.eval();var_wo_ddp.eval();text_encoder.eval()
    # for item in batched_iterator(coco_dataset,batch_size=16,select_size=select_sz):
    for item in prompt_with_id_list:
        prompt_id = [item["id"]]*sample_per_batch
        text_prompts = [item["prompt"]]*sample_per_batch

        recon_B3HW_=[]
        for i in range(2):
            text_prompt=text_prompts[i*5:(i+1)*5]
            B_=len(text_prompt)
            with torch.no_grad():
                with torch.inference_mode():
                    prompt_embeds,prompt_attention_mask,pooled_embed = text_encoder.extract_text_features(text_prompt+[""]*B_)
                    # with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                    recon_B3HW = var_wo_ddp.autoregressive_infer_cfg(B=B_, label_B=None, 
                                                                    encoder_hidden_states=prompt_embeds,
                                                                    encoder_attention_mask=prompt_attention_mask,
                                                                    encoder_pool_feat=pooled_embed,
                                                                    cfg=4.0, top_k=600,
                                                                    top_p=0.8,
                                                                    more_smooth=False)#smooth会导致细节减少，但可靠性提升
                    recon_B3HW_.append(recon_B3HW)
        recon_B3HW=torch.cat(recon_B3HW_,dim=0)
            
        for i,label in enumerate(prompt_id):
            img_pred = (recon_B3HW[i].permute(1, 2, 0).cpu()*255.).numpy().astype(np.uint8)#(hwc)
            Image.fromarray(img_pred).save(os.path.join(savedir_pred,f"{label}_{i}.png"))
            print(os.path.join(savedir_pred,f"{label}_{i}.png"),' evaluated and saved...')


if __name__=="__main__":
    save_root='/home/disk2/mxx/datasets/image_reward/benchmark-generations'
    savedir_pred=osp.join(save_root,'var_new_quality_tuned')
    if not os.path.exists(savedir_pred):os.makedirs(savedir_pred)

    prepare_images(savedir_pred=savedir_pred)


    # dataset=PairedImageDataset(root_pred=savedir_pred,root_ref=savedir_gt,transform=transform_image_fid)
    # loader=DataLoader(dataset,batch_size=20,shuffle=False)


    # # engine = Engine(dummy_process_function)
    # fid_metric = FID(device="cuda")  # 或者 "cuda" 如果可用

    # for i,data in enumerate(loader):
    #     # pdb.set_trace()
    #     fid_metric.update(data)
    #     # state=engine.run([data])
    #     # print("FID score: ", fid_metric.compute())
    # print("FID score: ", fid_metric.compute())




