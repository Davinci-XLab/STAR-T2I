import sys
sys.path.insert(0,'../../../')
from diffusers import DiffusionPipeline
from dataset.test_fid_coco import batched_iterator_MJHQ
import argparse
import random
import torch
from PIL import Image
import os.path as osp
import numpy as np
import os



def prepare_images(savedir_pred,savedir_gt,category):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    # repo_id = "/home/disk2/mxx/ckpts/playground-v2-1024px-aesthetic"
    repo_id="/nfs-26/maxiaoxiao/ckpts/playground-v2.5-1024px-aesthetic"
    print('loaded pipe...')
    pipe = DiffusionPipeline.from_pretrained(
        repo_id, use_safetensors=True,
            torch_dtype=torch.float16,
            add_watermarker=False
    ).to(device)



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

        oup_B3HW=pipe(text_prompts).images
            
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
    '''
    save_root='/nfs-141/maxiaoxiao/eval_results/compare_methods/FID_mjhq_1024/'
    savedir_pred=osp.join(save_root,'prediction_playground2.5_aes')
    savedir_gt=osp.join(save_root,'reference')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        default="",
        type=str
    )
    args = parser.parse_args()

    prepare_images(savedir_pred=savedir_pred,savedir_gt=savedir_gt,category=[args.category])

'''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    # # repo_id = "/home/disk2/mxx/ckpts/playground-v2-1024px-aesthetic"
    # repo_id="/home/disk2/nfs/maxiaoxiao/ckpts/playground-v2.5-1024px-aesthetic"
    # print('loaded pipe...')
    # text_prompts='A gorgeous woman with wavy black hair walking with a handsome man in suit, looking into each others eyes, with wisteria flowers in the background'
    # pipe = DiffusionPipeline.from_pretrained(
    #     repo_id, use_safetensors=True,
    #         torch_dtype=torch.float16,
    #         add_watermarker=False
    # ).to(device)
    # oup_B3HW=pipe(text_prompts).images[0]
    # oup_B3HW.save('playgroundv2.png')

    # from diffusers import PixArtAlphaPipeline
    # # repo_id = "/home/disk2/mxx/ckpts/playground-v2-1024px-aesthetic"
    # repo_id="/home/disk2/nfs/maxiaoxiao/ckpts/PixArt-XL-2-1024-MS"
    # print('loaded pipe...')
    # text_prompts='A gorgeous woman with wavy black hair walking with a handsome man in suit, looking into each others eyes, with wisteria flowers in the background'
    # pipe = PixArtAlphaPipeline.from_pretrained(
    #     repo_id, use_safetensors=True,
    #         torch_dtype=torch.float16,
    #         add_watermarker=False
    # ).to(device)
    # oup_B3HW=pipe(text_prompts).images[0]
    # oup_B3HW.save('pixart.png')


    # from diffusers import StableDiffusionXLPipeline
    # # repo_id = "/home/disk2/mxx/ckpts/playground-v2-1024px-aesthetic"
    # # repo_id = "/home/disk2/mxx/ckpts/playground-v2-1024px-aesthetic"
    # repo_id="/home/nfs/nfs-27/mowenyi/ckpt/ckpt/stable-diffusion-xl-base-1.0"
    # print('loaded pipe...')
    # text_prompts='A gorgeous woman with wavy black hair walking with a handsome man in suit, looking into each others eyes, with wisteria flowers in the background'
    # base = StableDiffusionXLPipeline.from_pretrained(
    #     repo_id,
    #     use_safetensors=True,
    #     torch_dtype=torch.float16
    # ).to(device)
    # oup_B3HW=base(text_prompts).images[0]
    # oup_B3HW.save('sdxl.png')

    # from diffusers import StableDiffusionPipeline
    # # repo_id = "/home/disk2/mxx/ckpts/playground-v2-1024px-aesthetic"
    # # repo_id = "/home/disk2/mxx/ckpts/playground-v2-1024px-aesthetic"
    # repo_id="/home/nfs/nfs-27/zhangguiwei/zhangguiwei/stable-diffusion-2-1"
    # print('loaded pipe...')
    # text_prompts='A gorgeous woman with wavy black hair walking with a handsome man in suit, looking into each others eyes, with wisteria flowers in the background'
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     repo_id, 
    #         torch_dtype=torch.float16,
    #         add_watermarker=False
    # ).to(device)
    # oup_B3HW=pipe(text_prompts).images[0]
    # oup_B3HW.save('sd21.png')


    from diffusers import PixArtAlphaPipeline
    repo_id="/home/disk2/nfs/maxiaoxiao/ckpts/PixArt-XL-2-1024-MS"
    print('loaded pipe...')
    text_prompts='Close up of an attractive bride, happy and smiling, looking up and in the distance in wonder, vibrant neon lighting, background out of focus'
    pipe = PixArtAlphaPipeline.from_pretrained(
        repo_id, 
            torch_dtype=torch.float16,
            add_watermarker=False
    ).to(device)
    oup_B3HW=pipe(text_prompts).images[0]
    oup_B3HW.save('pixart.png')

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




