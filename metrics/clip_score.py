import os
from torch import autocast, nn
import clip
import torch
from PIL import Image
import numpy as np


def clip_iterator(img_root,cap_root, batch_size):
    images = []
    captions = []
    img_files=os.listdir(img_root)
    for item in img_files:
        images.append(Image.open(os.path.join(img_root,item)))
        cap=open(os.path.join(cap_root,item.replace('.png','.txt')),'r').readline().strip()
        captions.append(cap)

        if len(captions) == batch_size:
            yield images, captions
            # 清空列表以准备下一个批次
            images = []
            captions = []
    # 处理剩余的items，如果有的话
    if images:
        yield images, captions



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


def get_clip_features(clip_model, clip_preprocess, pil_image):

    images = [clip_preprocess(i) for i in pil_image]
    image = torch.stack(images)
    
    image = image.to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    return image_features



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_root=str(input('input image path:'));print('img root=',img_root)
    cap_root=str(input('input cap path:'));print('cap root=',cap_root)
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device, jit=False)
    # clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    all_scores=[]
    for images,prompts in clip_iterator(img_root=img_root,cap_root=cap_root,batch_size=16):
        image_features = get_clip_features(clip_model, clip_preprocess, images)
        clip_scores = get_clip_score_batched(clip_model ,image_features, prompts,device)
        all_scores=all_scores+clip_scores
        print(clip_scores)
    final=np.mean(np.array(all_scores))
    print('length=',len(all_scores),' final score=',final)