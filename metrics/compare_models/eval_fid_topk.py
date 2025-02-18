import os
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


class PairedImageDataset(Dataset):
    def __init__(self, root_pred, root_ref, transform=None):
        self.root_pred = root_pred
        self.root_ref = root_ref
        self.transform = transform
        self.filenames = os.listdir(root_pred)  # 假设预测和参考文件夹中的文件完全匹配

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        pred_path = os.path.join(self.root_pred, self.filenames[idx])
        ref_path = os.path.join(self.root_ref, self.filenames[idx])

        pred_image = Image.open(pred_path).convert('RGB')
        ref_image = Image.open(ref_path).convert('RGB')

        if self.transform:
            pred_image = self.transform(pred_image)
            ref_image = self.transform(ref_image)

        return pred_image, ref_image



def transform_image_fid(image):
    transform=transforms.Compose([
        transforms.Resize(299),  # 等比缩放图像，使最小边为256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


def prepare_images(savedir_pred,savedir_gt, gen_reso=256, topk=600, depth=30,batch_sz=8,w_mask=False):
    # 一些参数
    select_sz=batch_sz
    
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)#, 24, 32)
    texenc_ckpt='/home/disk2/nfs/maxiaoxiao/ckpts/stable-diffusion-2-1'
    vae_ckpt='/home/disk2/nfs/maxiaoxiao/ckpts/FoundationVision-var/vae_ch160v4096z32.pth'
    var_ckpt='/home/nfs/nfs-141/maxiaoxiao/workspace/var_rope_d16_theta_modified_anorm/ar-ckpt-ep7-iter0.pth'

    if not osp.exists(savedir_pred):os.makedirs(savedir_pred)
    if not osp.exists(savedir_gt):os.makedirs(savedir_gt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if gen_reso==256:
        patch_nums =(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        if depth==16:
            var_ckpt='/home/nfs/nfs-141/maxiaoxiao/workspace/var_rope_d16_theta_modified_anorm/ar-ckpt-ep7-iter0.pth'
        else:
            var_ckpt='/home/nfs/nfs-142/maxiaoxiao/workspace/var_rope_d30_256_stage2/ar-ckpt-ep3-iter10000.pth'
        enable_logit_norm=True
    elif gen_reso==512:
        patch_nums =[1, 2, 3, 4, 6, 9, 13, 18, 24, 32]
        var_ckpt='/home/disk2/nfs/maxiaoxiao/workspace/var_rope_d30_512_quality_tuning/ar-ckpt-ep4-iter75000.pth'
        enable_logit_norm=False
    elif gen_reso==1024:
        patch_nums =[1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64]
                   #[1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64],
        # var_ckpt='/home/nfs/nfs-142/maxiaoxiao/workspace/var_rope_d30_1024_drop_3/ar-ckpt-ep0-iter57000.pth'
        # var_ckpt='/home/nfs/nfs-142/maxiaoxiao/workspace/var_rope_d30_1024_vqa/ar-ckpt-last.pth'
        # var_ckpt='/home/nfs/nfs-142/maxiaoxiao/workspace/var_rope_d30_1024_vqa_tune/ar-ckpt-ep0-iter3000.pth'
        var_ckpt='/home/nfs/nfs-142/maxiaoxiao/workspace/var_rope_d30_1024_sampler_mask/ar-ckpt-ep1-iter26000.pth'
        var_ckpt='/home/nfs/nfs-132/liangtao/workspace/star/var_rope_d30_1024_sampler_mask/ar-ckpt-ep1-iter32000.pth'
        var_ckpt='/home/nfs/nfs-132/liangtao/workspace/star/var_rope_d30_1024_sampler_mask_0103/ar-ckpt-ep0-iter32000.pth'
        # var_ckpt='/home/nfs/nfs-142/maxiaoxiao/workspace/var_rope_d30_1024/ar-ckpt-ep0-iter72000.pth'
        enable_logit_norm=True


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

    var_wo_ddp.load_state_dict(torch.load(var_ckpt, map_location='cpu')["trainer"]["var_wo_ddp"], strict=True)
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    text_encoder,_=build_text(pretrained_path=texenc_ckpt,device=device)
    vae_local.eval();var_wo_ddp.eval();text_encoder.eval()

    # set args
    seed = 4 #@param {type:"number"}
    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    mjhq_root='/home/nfs/nfs-141/maxiaoxiao/eval_results/var_rope_final/d30_topk_mjhq/reference'
    #mjhq_root='/home/disk2/nfs/maxiaoxiao/datasets/MJHQ-30K'


    for item in batched_iterator_MJHQ(mjhq_root,batch_size=select_sz,select_size=select_sz,
                                      category=[],imsize=gen_reso):
        imgs_B3HW=item['image']
        text_prompts=item['caption']
        name=item['name']
        category=item['category']
        B_=len(text_prompts)
        print(text_prompts)
        print(B_,imgs_B3HW.shape,len(text_prompts),len(name),len(category))
        with torch.no_grad():
            with torch.inference_mode():
                prompt_embeds,prompt_attention_mask,pooled_embed = text_encoder.extract_text_features(text_prompts+[""]*B_)
                with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                    recon_B3HW = var_wo_ddp.autoregressive_infer_cfg(B=B_, label_B=None, 
                                                                    encoder_hidden_states=prompt_embeds,
                                                                    encoder_attention_mask=prompt_attention_mask,
                                                                    encoder_pool_feat=pooled_embed,
                                                                    cfg=4.0, top_k=topk,
                                                                    top_p=0.8,g_seed=seed,
                                                                    more_smooth=False,
                                                                    w_mask=w_mask,
                                                                    sample_version='1024')
            
        for i,label in enumerate(name):
            img_pred = (recon_B3HW[i].permute(1, 2, 0).cpu()*255.).numpy().astype(np.uint8)#(hwc)
            img_gt = (imgs_B3HW[i].permute(1, 2, 0).add_(1).mul_(0.5).clamp_(0,1).cpu()*255.).numpy().astype(np.uint8)#(hwc)

            cur_category=category[i]
            if not osp.exists(osp.join(savedir_pred,cur_category)):os.makedirs(osp.join(savedir_pred,cur_category))
            if not osp.exists(osp.join(savedir_gt,cur_category)):os.makedirs(osp.join(savedir_gt,cur_category))

            Image.fromarray(img_pred).save(osp.join(savedir_pred,cur_category,'%s.png'%label))

            # 保存原图
            Image.fromarray(img_gt).save(osp.join(savedir_gt,cur_category,'%s.png'%label))
            print(osp.join(savedir_pred,cur_category,'%s.png'%(label)),' evaluated and saved...')


if __name__=="__main__":
    import argparse

    # 创建解析器
    parser = argparse.ArgumentParser(description='Model parameters')

    # 添加参数
    parser.add_argument('--depth', type=int, default=30, help='Model depth')
    parser.add_argument('--save_root', type=str, default='/home/nfs/nfs-132/star/eval_results/var_rope_final/d30_1024_sampler_topk_mjhq_0106/', help='Path to save results')
    parser.add_argument('--gen_reso', type=int, default=1024, help='Image resolution')
    parser.add_argument('--batch_sz', type=int, default=6, help='batchsize')
    parser.add_argument('--topk', type=int, default=4096, help='Image resolution')
    parser.add_argument('--w_mask', type=str, default='true')
    args = parser.parse_args()
    args.w_mask=args.w_mask=='true'
    savedir_pred=osp.join(args.save_root,'1024gen/prediction_top%d_mask%s_v3'%(args.topk,args.w_mask))
    savedir_gt=osp.join(args.save_root,'1024gen/reference_3W')

    prepare_images(savedir_pred=savedir_pred,savedir_gt=savedir_gt,
                   gen_reso=args.gen_reso, topk=args.topk, w_mask=args.w_mask)


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




