import sys
sys.path.insert(0,'../../')
from dataset.test_fid_coco import batched_iterator,transform_image,batched_iterator_MJHQ
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


def zero_out_lvl_emb_weights(model):#呃因为之前的256版本的level emb没存下来，这样尽量得到一个合理的结果
    for name, param in model.named_parameters():
        # 检查参数名称是否与 Cross Attention 层的权重匹配
        # print(name,param.shape,param.max(),param.min())
        if ('lvl_embed.weight' in name):
            # 将权重置为0
            with torch.no_grad():  # 确保不会在这个操作中跟踪梯度
                param.fill_(0.0)
    print("lvl_embed.weight have been zeroed out.")


def prepare_images(savedir_pred,savedir_gt):
    # 一些参数
    select_sz=6#16
    
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    texenc_ckpt='/home/disk2/mxx/ckpts/stable-diffusion-2-1'
    vae_ckpt='/home/disk2/mxx/ckpts/FoundationVision-var/vae_ch160v4096z32.pth'
    var_ckpt='/home/disk1/maxiaoxiao/workspace/var_rope_d30_debug/ar-ckpt-ep19-iter0.pth'

    if not osp.exists(savedir_pred):os.makedirs(savedir_pred)
    if not osp.exists(savedir_gt):os.makedirs(savedir_gt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae_local, var_wo_ddp = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        depth=30, shared_aln=False, attn_l2_norm=False,
        in_dim_cross=1024,#TODO:换成从text enc得到的参数
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=5e-5, init_head=0.02, init_std=-1,
        rope_emb=True,lvl_emb=True,
    )

    missing,unexpected=var_wo_ddp.load_state_dict(torch.load(var_ckpt, map_location='cpu')["trainer"]["var_wo_ddp"], strict=False)
    print('missing: ',missing,'unexpected: ',unexpected)
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    text_encoder=build_text(pretrained_path=texenc_ckpt,device=device)

    # zero_out_lvl_emb_weights(var_wo_ddp)

    vae_local.eval();var_wo_ddp.eval();text_encoder.eval()

    # set args
    seed = 4 #@param {type:"number"}
    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # coco_dataset = hf_datasets.load_dataset("/home/disk2/mxx/datasets/coco-30-val-2014", split="train", streaming=True)
    # print(coco_dataset)

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    mjhq_root='/home/disk2/mxx/datasets/MJHQ-30K'


    for item in batched_iterator_MJHQ(mjhq_root,batch_size=6,select_size=select_sz,
                                      category=[],imsize=256):
        imgs_B3HW=item['image']
        text_prompts=item['caption']
        name=item['name']
        category=item['category']
        B_=len(text_prompts)
        print(text_prompts)
        print(B_,imgs_B3HW.shape,len(text_prompts),len(name),len(category))
        # print(item)
        with torch.no_grad():
            with torch.inference_mode():
                prompt_embeds,prompt_attention_mask,pooled_embed = text_encoder.extract_text_features(text_prompts+[""]*B_)
                # with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                recon_B3HW = var_wo_ddp.autoregressive_infer_cfg(B=B_, label_B=None, 
                                                                encoder_hidden_states=prompt_embeds,
                                                                encoder_attention_mask=prompt_attention_mask,
                                                                encoder_pool_feat=pooled_embed,
                                                                cfg=4.0, top_k=600,
                                                                top_p=0.8,g_seed=seed,
                                                                more_smooth=False)#smooth会导致细节减少，但可靠性提升
            
        for i,label in enumerate(name):
            img_pred = (recon_B3HW[i].permute(1, 2, 0).cpu()*255.).numpy().astype(np.uint8)#(hwc)
            img_gt = (imgs_B3HW[i].permute(1, 2, 0).add_(1).mul_(0.5).clamp_(0,1).cpu()*255.).numpy().astype(np.uint8)#(hwc)

            cur_category=category[i]
            if not osp.exists(osp.join(savedir_pred,cur_category)):os.makedirs(osp.join(savedir_pred,cur_category))
            if not osp.exists(osp.join(savedir_gt,cur_category)):os.makedirs(osp.join(savedir_gt,cur_category))

            Image.fromarray(img_pred).save(osp.join(savedir_pred,cur_category,'%s.png'%label))

            # 保存原图
            Image.fromarray(img_gt).save(osp.join(savedir_gt,cur_category,'%s.png'%label))
            print(osp.join(savedir_gt,cur_category,'%s.png'%(label)),' evaluated and saved...')


if __name__=="__main__":
    save_root='/home/disk1/maxiaoxiao/workspace/var_rope_d30_debug/FID_mjhq'
    savedir_pred=osp.join(save_root,'prediction')
    savedir_gt=osp.join(save_root,'reference')

    prepare_images(savedir_pred=savedir_pred,savedir_gt=savedir_gt)


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




