# import datasets as hf_datasets
from torchvision import transforms
import pdb
import numpy as np
import random
import torch
import json
import os
from PIL import Image


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def transform_image(image,size=512):
    transform=transforms.Compose([
        transforms.Resize(size),  # 等比缩放图像，使最小边为256
        transforms.RandomCrop(size),  # 随机裁剪出256x256的图像区域
        transforms.ToTensor(), normalize_01_into_pm1,
    ])
    return transform(image)


def batched_iterator_MJHQ(data_root, batch_size, select_size, category=[], seed=42, imsize=512):
    random.seed(seed)  # 设置随机种子以确保可复现性
    with open(os.path.join(data_root,'meta_data.json'),'r') as f: meta_json=json.load(f)
    if len(category)>0:
        meta_json={key:value for key,value in meta_json.items() if value['category'] in category}

    images = []
    captions = []
    names=[]
    categories=[]
    for item_name,item in meta_json.items():
        impath=os.path.join(data_root,'data_meta',item['category'],'%s.jpg'%item_name)
        images.append(transform_image(Image.open(impath),size=imsize))
        captions.append(item['prompt'])
        names.append(item_name)
        categories.append(item['category'])

        if len(captions) == batch_size:
            # 当达到batch_size时，随机选择select_size个item
            indices = random.sample(range(batch_size), select_size)
            selected_images = [images[i] for i in indices]
            selected_captions = [captions[i] for i in indices]
            selected_names = [names[i] for i in indices]
            selected_categories = [categories[i] for i in indices]
            yield {'image': torch.stack(selected_images, dim=0), 'caption': selected_captions, 'name':selected_names, 'category':selected_categories}

            # 清空列表以准备下一个批次
            images = []
            captions = []
            names = []
            categories = []

    # 处理剩余的items，如果有的话
    if images:
        # 如果剩余项的数量大于或等于select_size，进行选择
        if len(images) >= select_size:
            indices = random.sample(range(len(images)), select_size)
            selected_images = [images[i] for i in indices]
            selected_captions = [captions[i] for i in indices]
            selected_names = [names[i] for i in indices]
            selected_categories = [categories[i] for i in indices]
            yield {'image': torch.stack(selected_images, dim=0), 'caption': selected_captions, 'name':selected_names, 'category':selected_categories}
        else:
            # 如果不足select_size，则返回所有剩余项
            yield {'image': torch.stack(images, dim=0), 'caption': captions, 'name':names, 'category':categories}


def batched_iterator(dataset, batch_size, select_size, seed=42):
    random.seed(seed)  # 设置随机种子以确保可复现性
    images = []
    captions = []
    for item in dataset:
        if item['image'].mode == 'RGB':  # 只处理彩色图片
            images.append(transform_image(item['image']))
            captions.append(item['caption'])

            if len(captions) == batch_size:
                # 当达到batch_size时，随机选择select_size个item
                indices = random.sample(range(batch_size), select_size)
                selected_images = [images[i] for i in indices]
                selected_captions = [captions[i] for i in indices]

                yield {'image': torch.stack(selected_images, dim=0), 'caption': selected_captions}

                # 清空列表以准备下一个批次
                images = []
                captions = []

    # 处理剩余的items，如果有的话
    if images:
        # 如果剩余项的数量大于或等于select_size，进行选择
        if len(images) >= select_size:
            indices = random.sample(range(len(images)), select_size)
            selected_images = [images[i] for i in indices]
            selected_captions = [captions[i] for i in indices]
            yield {'image': torch.stack(selected_images, dim=0), 'caption': selected_captions}
        else:
            # 如果不足select_size，则返回所有剩余项
            yield {'image': torch.stack(images, dim=0), 'caption': captions}


if __name__=="__main__":
    # ---------------------load coco------------------------------
    # coco_dataset = hf_datasets.load_dataset("/home/disk2/mxx/datasets/coco-30-val-2014", split="train", streaming=True)
    # print(coco_dataset)

    # for item in batched_iterator(coco_dataset,batch_size=20,select_size=2):
    #     # print(item)
    #     pass

    # ----------------------mjhq----------------------------------
    mjhq_root='/home/disk2/mxx/datasets/MJHQ-30K'

    for item in batched_iterator_MJHQ(mjhq_root,batch_size=20,select_size=2):
        # print(item)
        pass
