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


def prepare_images(savedir_pred,savedir_gt,mode=''):
    # 一些参数
    
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    texenc_ckpt='/home/disk2/mxx/ckpts/stable-diffusion-2-1'
    vae_ckpt='/home/disk2/mxx/ckpts/FoundationVision-var/vae_ch160v4096z32.pth'

    if not osp.exists(savedir_pred):os.makedirs(savedir_pred)
    if not osp.exists(savedir_gt):os.makedirs(savedir_gt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if mode=='full':
        var_ckpt='/home/disk2/mxx/workspace/var_rope_ablations_d16/full_ep15_iter0.pth'
        vae_local, var_wo_ddp = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
            device=device, patch_nums=patch_nums,
            depth=16, shared_aln=False, attn_l2_norm=False,
            in_dim_cross=1024,#TODO:换成从text enc得到的参数
            flash_if_available=True, fused_if_available=True,
            init_adaln=0.5, init_adaln_gamma=5e-5, init_head=0.02, init_std=-1,
            rope_emb=True,lvl_emb=True,
        )
    elif mode=='ab_cross':
        var_ckpt='/home/disk2/mxx/workspace/var_rope_ablations_d16/ab_cross_ep15_iter0.pth'
        vae_local, var_wo_ddp = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
            device=device, patch_nums=patch_nums,
            depth=16, shared_aln=False, attn_l2_norm=False,
            enable_cross=False,
            in_dim_cross=1024,#TODO:换成从text enc得到的参数
            flash_if_available=True, fused_if_available=True,
            init_adaln=0.5, init_adaln_gamma=5e-5, init_head=0.02, init_std=-1,
            rope_emb=True,lvl_emb=True,
        )
    elif mode=='ab_rope':
        var_ckpt='/home/disk2/mxx/workspace/var_rope_ablations_d16/ab_rope_ep15_iter0.pth'
        vae_local, var_wo_ddp = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
            device=device, patch_nums=patch_nums,
            depth=16, shared_aln=False, attn_l2_norm=False,
            enable_cross=True,
            in_dim_cross=1024,#TODO:换成从text enc得到的参数
            flash_if_available=True, fused_if_available=True,
            init_adaln=0.5, init_adaln_gamma=5e-5, init_head=0.02, init_std=-1,
            rope_emb=False,lvl_emb=True,
        )


    # set args
    seed = 42 #@param {type:"number"}
    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    var_wo_ddp.load_state_dict(torch.load(var_ckpt, map_location='cpu')["trainer"]["var_wo_ddp"], strict=True)
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    text_encoder=build_text(pretrained_path=texenc_ckpt,device=device)
    vae_local.eval();var_wo_ddp.eval();text_encoder.eval()

    # coco_dataset = hf_datasets.load_dataset("/home/disk2/mxx/datasets/coco-30-val-2014", split="train", streaming=True)
    # print(coco_dataset)

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    mjhq_root='/home/disk2/mxx/datasets/MJHQ-30K'


    for item in batched_iterator_MJHQ(mjhq_root,batch_size=12,select_size=12,
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
    mode=input('mode:')#full,ab_cross,ab_rope
    save_root='/home/disk2/mxx/workspace/var_rope_d16_ab_256/FID_mjhq'
    savedir_pred=osp.join(save_root,'prediction_%s'%mode)
    savedir_gt=osp.join(save_root,'reference')

    prepare_images(savedir_pred=savedir_pred,savedir_gt=savedir_gt,mode=mode)


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




