import sys
sys.path.insert(0,'../../')

from models import VAR, VQVAE, build_vae_var
from models.text_encoder import build_text
from utils.utils import format_sentence
import random
import torch
from PIL import Image
import os.path as osp
import pdb
import numpy as np
import os
import json
from utils.utils import format_sentence

def read_instance_prompts(prompt_path='/home/disk2/mxx/VAR/dataset/prompts/benchmark_v3_en.txt'):
    prompts = []
    with open(prompt_path) as prompt_file:
        for line in prompt_file.readlines():
            prompts.append(line.strip())
    return prompts


def split_list(input_list, length):
    return [input_list[i:i+length] for i in range(0, len(input_list), length)]

def prepare_images(savedir_pred,sample_per_batch=10):
    # 一些参数
    select_sz=1#16

    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark=False

    prompt_with_id_list=read_instance_prompts()
    
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    # patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16,24,32)
    # texenc_ckpt='/nfs-26/maxiaoxiao/ckpts/stable-diffusion-2-1'
    # vae_ckpt='/nfs-26/maxiaoxiao/ckpts/FoundationVision-var/vae_ch160v4096z32.pth'
    # var_ckpt='/nfs-26/maxiaoxiao/workspace/var_rope_d30_lvl_scratch_ln/ar-ckpt-ep12-iter20000.pth'
    texenc_ckpt='/home/disk2/mxx/ckpts/stable-diffusion-2-1'
    vae_ckpt='/home/disk2/mxx/ckpts/FoundationVision-var/vae_ch160v4096z32.pth'
    # var_ckpt='/home/disk2/mxx/workspace/var_rope_d30_512/ar-ckpt-ep2-iter50000.pth'
    var_ckpt='/home/disk2/mxx/workspace/var_rope_ablations_d16/full_ep15_iter0.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae_local, var_wo_ddp = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        depth=16, shared_aln=False, attn_l2_norm=False,
        enable_cross=True,
        in_dim_cross=1024,#TODO:换成从text enc得到的参数
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=5e-5, init_head=0.02, init_std=-1,
        rope_emb=True,lvl_emb=True,
    )

    try:
        var_wo_ddp.load_state_dict(torch.load(var_ckpt, map_location='cpu')["trainer"]["var_wo_ddp"], strict=True)
    except:
        print('unsing strict=False in loading...')
        missing, unexpected=var_wo_ddp.load_state_dict(torch.load(var_ckpt, map_location='cpu')["trainer"]["var_wo_ddp"], strict=False)
        # var_wo_ddp.lvl_1LC=torch.zeros([1,2280,1920]).to(device)
        print(missing,unexpected)
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

    text_encoder=build_text(pretrained_path=texenc_ckpt,device=device)
    # text_encoder.eval()
    # text_encoder.load_state_dict(torch.load(var_ckpt, map_location='cpu')["text_enc"], strict=False)

    # text_encoder.text_encoder.eval()
    # for p in text_encoder.text_encoder.parameters():p.requires_grad_(False)
    # for p in vae_local.parameters():p.requires_grad_(False)
    # for p in var_wo_ddp.parameters():p.requires_grad_(False)
    # for name, param in text_encoder.named_parameters():
    #     param.requires_grad_(False)
        

    # run faster
    tf32 = False
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # set args
    seed = 4 #@param {type:"number"}
    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    vae_local.eval();var_wo_ddp.eval();text_encoder.eval()
    # for name, param in text_encoder.named_parameters():
    #     print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    # for name, param in vae_local.named_parameters():
    #     print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    # for name, param in var_wo_ddp.named_parameters():
    #     print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")


    prompt_with_id_list=split_list(prompt_with_id_list,length=4)
    for text_prompts in prompt_with_id_list:
        B_=len(text_prompts)
        
        with torch.no_grad():
            with torch.inference_mode():
                prompt_embeds,prompt_attention_mask,pooled_embed = text_encoder.extract_text_features(text_prompts+[""]*B_)
                with torch.autocast('cuda', cache_enabled=True,dtype=torch.float32): 
                    recon_B3HW = var_wo_ddp.autoregressive_infer_cfg(B=B_, label_B=None, 
                                                                    encoder_hidden_states=prompt_embeds,
                                                                    encoder_attention_mask=prompt_attention_mask,
                                                                    encoder_pool_feat=pooled_embed,
                                                                    cfg=4.0, top_k=600,
                                                                    top_p=0.8,g_seed=seed,
                                                                    more_smooth=True)#smooth会导致细节减少，但可靠性提升
            
        for i,item in enumerate(text_prompts):
            img_pred = (recon_B3HW[i].permute(1, 2, 0).cpu()*255.).numpy().astype(np.uint8)#(hwc)
            Image.fromarray(img_pred).save(osp.join(savedir_pred,'%s.png'%(format_sentence(item))))
            print(item,' evaluated and saved...')


if __name__=="__main__":
    save_root='./test/'
    savedir_pred=osp.join(save_root,'var_local')
    if not os.path.exists(savedir_pred):
        os.makedirs(savedir_pred)

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




