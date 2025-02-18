# coding=utf-8
import os
import cv2
import json
from glob import glob
from tqdm import tqdm


def common_convertor(caption_path, data_dir, image_dir_name, json_dir_name):
    json_dir = os.path.join(data_dir, json_dir_name)
    os.makedirs(json_dir, exist_ok=True)
    image_dir = os.path.join(data_dir, image_dir_name)
    paths = glob(os.path.join(image_dir, '*.*g'))

    captions = {}
    with open(caption_path, 'r') as file:
        for line in file.readlines():
            line = line.strip('\n')
            basename, caption = line.split('\t')
            short_name = os.path.splitext(basename)[0]  # 后缀可能发生改变，前缀保持不变
            captions[short_name] = caption

    progress_bar = tqdm(range(len(paths)), total=len(paths))
    progress_bar.set_description("Processing")

    for i, path in enumerate(paths):
        try:
            basename = os.path.basename(path)
            short_name = os.path.splitext(basename)[0]
            image = cv2.imread(path)
            h, w, c = image.shape
            if c != 3:
                continue
            info = {}
            id = short_name + '-{}'.format(i)
            prompt = captions.get(short_name, None)
            if prompt is None:
                raise ValueError('The prompt is None.')
            height = h
            width = w

            info['id'] = id
            info['basename'] = basename
            info['prompt'] = prompt
            info['height'] = height
            info['width'] = width
            json_path = os.path.join(json_dir, short_name + '.json')
            json.dump(info, open(json_path, "w"), indent=4)
        except Exception as e:
            print(e)
        progress_bar.update(1)
