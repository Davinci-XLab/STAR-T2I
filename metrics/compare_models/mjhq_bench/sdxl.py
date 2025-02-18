import sys
sys.path.insert(0,'../../../')
from diffusers import StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline,AutoencoderKL,EulerAncestralDiscreteScheduler
from dataset.test_fid_coco import batched_iterator_MJHQ
import argparse
import random
import torch
from PIL import Image
import os.path as osp
import pdb
import numpy as np
import os



def prepare_images(savedir_pred,savedir_gt):
    # 一些参数
    select_sz=8#16

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base = StableDiffusionXLPipeline.from_pretrained(
        "/nfs-26/maxiaoxiao/ckpts/stable-diffusion-xl-base-1.0",
        use_safetensors=True,
        torch_dtype=torch.float16
    ).to(device)
    print('loaded pipe...')


    mjhq_root='/nfs-26/maxiaoxiao/datasets/MJHQ-30K'


    # set args
    seed = 4 #@param {type:"number"}
    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


    for item in batched_iterator_MJHQ(mjhq_root,batch_size=4,select_size=4,
                                      category=category):
        imgs_B3HW=item['image']
        text_prompts=item['caption']
        name=item['name']
        category=item['category']

        oup_B3HW=base(text_prompts).images
            
        for i,label in enumerate(name):
            img_gt = (imgs_B3HW[i].permute(1, 2, 0).add_(1).mul_(0.5).clamp_(0,1).cpu()*255.).numpy().astype(np.uint8)#(hwc)

            cur_category=category[i]
            if not osp.exists(osp.join(savedir_pred,cur_category)):os.makedirs(osp.join(savedir_pred,cur_category))
            if not osp.exists(osp.join(savedir_gt,cur_category)):os.makedirs(osp.join(savedir_gt,cur_category))

            oup_B3HW[i].save(osp.join(savedir_pred,cur_category,'%s.png'%label))

            # 保存原图
            Image.fromarray(img_gt).save(osp.join(savedir_gt,cur_category,'%s.png'%label))
            print(osp.join(savedir_pred,cur_category,'%s.png'%(label)),' evaluated and saved...')


if __name__=="__main__":
    save_root='/nfs-141/maxiaoxiao/eval_results/compare_methods/FID_mjhq_1024/'
    savedir_pred=osp.join(save_root,'prediction_sdxl')
    savedir_gt=osp.join(save_root,'reference')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        default="",
        type=str
    )
    args = parser.parse_args()

    prepare_images(savedir_pred=savedir_pred,savedir_gt=savedir_gt,category=[args.category])

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




