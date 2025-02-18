import os
from torch import autocast, nn
import clip
import torch
from PIL import Image
import json
import numpy as np

import os
import json
import random
from tqdm import tqdm 

def clip_iterator_mjhq(img_root,cap_root, batch_size, category=[], seed=42):
    random.seed(seed)  # 设置随机种子以确保可复现性
    # 读取元数据文件
    try:
        with open(os.path.join(cap_root, 'meta_mjhq.json'), 'r') as f:
            meta_json = json.load(f)
    except:
        with open(os.path.join(cap_root, 'meta_data.json'), 'r') as f:
            meta_json = json.load(f)
    
    # 如果指定了类别，过滤出相应的条目
    if len(category) > 0:
        meta_json = {key: value for key, value in meta_json.items() if value['category'] in category}

    captions = []
    names = []
    categories = []
    images=[]
    # 迭代处理每个项目
    for item_name, item in meta_json.items():
        try:
            image=Image.open(os.path.join(img_root, item['category'], f"{item_name}.png"))
            images.append(image)
            captions.append(item['prompt'])
            names.append(item_name)
            categories.append(item['category'])
        except:
            pass

        # 当积累到足够的batch_size时，生成一批数据
        if len(captions) == batch_size:
            yield {'caption': captions, 'name': names, 'category': categories,'image':images}
            
            # 清空列表以准备下一个批次
            captions = []
            names = []
            categories = []
            images=[]

    # 处理最后一批剩余的items（如果有的话）
    if captions:
        yield {'caption': captions, 'name': names, 'category': categories,'image':images}


def get_clip_score_batched(clip_model, image_features, prompts,device):
    
    tokens = clip.tokenize(prompts, truncate=True).to(device)

    with torch.no_grad():
        if len(image_features) != len(prompts):
            assert len(image_features) % len(prompts) == 0
            tokens = (
                tokens.unsqueeze(1)
                .expand(-1, 1, -1)
                .reshape(-1, tokens.shape[-1])
            )

        text_features = clip_model.encode_text(tokens)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit = image_features @ text_features.t()
    scores = logit.diag().tolist()
    return scores


def get_clip_features(clip_model, prompts, clip_preprocess, pil_image):
    images=[]
    prompts_=[]
    for i in range(len(pil_image)):
        try:
            images.append(clip_preprocess(pil_image[i]))
            prompts_.append(prompts[i])
        except:
            pass

    image = torch.stack(images)
    
    image = image.to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    return image_features,prompts_



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # img_root=str(input('input image path:'));print('img root=',img_root)
    # cap_root=str(input('input cap path:'));print('cap root=',cap_root)
    img_root = '/home/nfs/nfs-141/liangtao/eval_results/var_rope_final/d30-1024_FID_mjhq/prediction'
    img_root = '/home/nfs/nfs-142/liangtao/eval_results/var_rope_final/d30-1024_FID_mjhq/prediction'
    img_root = '/home/nfs/nfs-142/liangtao/eval_results/var_rope_final/d30_1024_sampler_mjhq/1024gen/prediction_top4096'
    img_root = '/home/nfs/nfs-132/star/eval_results/var_rope_final/d30_1024_sampler_topk_mjhq/1024gen/prediction_top4096_maskTrue_v3'
    img_root = '/home/nfs/nfs-132/star/eval_results/var_rope_retrain1219/d30-256_FID_mjhq_ep1_20000/prediction'
    # img_root = '/home/nfs/nfs-132/star/eval_results/var_rope_final/d30-512_FID_mjhq/prediction'
    cap_root = '/home/disk2/nfs/maxiaoxiao/datasets/MJHQ-30K'
    clip_model, clip_preprocess = clip.load("/home/nfs/nfs-141/maxiaoxiao/ckpts/CLIP_ViT/ViT-L-14.pt", device=device, jit=False)
    # clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    all_scores=[]
    for item in tqdm(clip_iterator_mjhq(img_root=img_root,cap_root=cap_root,batch_size=32)):
        images=item['image']
        prompts=item['caption']
        image_features,prompts = get_clip_features(clip_model, prompts,clip_preprocess, images)
        clip_scores = get_clip_score_batched(clip_model ,image_features, prompts,device)
        all_scores=all_scores+clip_scores
        # print(clip_scores)
    final=np.mean(np.array(all_scores))
    print('length=',len(all_scores),' final score=',final)