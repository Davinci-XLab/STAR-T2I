import os
import json
import random
import torch
from PIL import Image

def transform_image(image, size=512):
    # 这里是你的图像处理逻辑
    return image.resize((size, size))

def batched_iterator_MJHQ(data_root, json_path, save_root, select_size=300, category=[], seed=42, imsize=512):
    random.seed(seed)  # 设置随机种子以确保可复现性
    with open(os.path.join(json_path, 'meta_mjhq.json'), 'r') as f:
        meta_json = json.load(f)

    if len(category) > 0:
        meta_json = {key: value for key, value in meta_json.items() if value['category'] in category}

    selected_images = []
    selected_meta = {}

    for cat in set(value['category'] for value in meta_json.values()):
        images = []
        for item_name, item in meta_json.items():
            if item['category'] == cat:
                impath = os.path.join(data_root, 'data_meta', item['category'], '%s.jpg' % item_name)
                images.append((item_name, Image.open(impath), item))

        # 随机选择300张图
        selected = random.sample(images, min(select_size, len(images)))
        selected_images.extend(selected)

        # 生成新meta信息
        for item_name, img, item in selected:
            selected_meta[item_name] = item

        # 保存图片到新文件夹
        category_folder = os.path.join(save_root,'data_meta', cat)
        os.makedirs(category_folder, exist_ok=True)
        for item_name, img, _ in selected:
            img.save(os.path.join(category_folder, f"{item_name}.jpg"))

    # 保存新meta信息到json文件
    with open(os.path.join(save_root, 'meta_mjhq.json'), 'w') as f:
        json.dump(selected_meta, f, indent=4)

# 使用示例
data_root='/home/disk2/nfs/maxiaoxiao/datasets/MJHQ-30K'
json_path='/home/disk2/nfs/maxiaoxiao/VAR'
save_root = '/home/nfs/nfs-141/maxiaoxiao/eval_results/var_rope_final/d30_topk_mjhq/reference'  # 替换为你的保存路径
batched_iterator_MJHQ(data_root, json_path, save_root)
