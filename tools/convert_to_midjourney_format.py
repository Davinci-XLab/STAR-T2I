# coding=utf-8

import os
import webdataset as wds
import json
from glob import glob
from tqdm import tqdm
from tools import multi_process

""" 
翻译之前，先将其他数据格式的 json 文件转为 midjourney 格式
"""


def convert_laion_core(data_dir, paths):
    progress_bar = tqdm(range(len(paths)), total=len(paths))
    progress_bar.set_description("Processing")

    for path in paths:
        basename = os.path.basename(path)
        short_name = os.path.splitext(basename)[0]
        out_path = os.path.join(data_dir, '{}_info.json'.format(short_name))
        if os.path.exists(out_path):
            pass
        else:
            res = {}
            count = 0
            try:
                dataset = wds.WebDataset(path)
                for sample in dataset:  # 遍历 tar 中的每个 json 文件
                    json_bytes = sample['json']
                    json_str = str(json_bytes, 'utf-8')  # 从二进制中恢复字符串
                    json_ = json.loads(json_str)

                    # sha256 = json_['sha256']  # 取出唯一标识
                    key = json_['key']
                    json_['prompt'] = json_['caption']

                    if key in res:
                        raise ValueError('The sha256 has repeat {}.'.format(key))
                    res[key] = json_
                    count += 1
            except Exception as e:
                print(e)
                print('Error path : {}'.format(path))

            if len(res) == 0:
                pass
            else:
                with open(out_path, 'w') as file:
                    file.write(json.dumps(res, indent=4, ensure_ascii=False))
        progress_bar.update(1)


def convert_laion(data_dir, num_work=20):
    paths = glob(os.path.join(data_dir, '*.tar'))
    print('Total handle: {}'.format(len(paths)))
    if num_work <= 1:
        convert_laion_core(data_dir=data_dir, paths=paths)
    else:
        num_work = min(len(paths), num_work)
        paths_ = multi_process.assign_task(paths, num_work)
        multi_process.start_task(
            target_func=convert_laion_core,
            args=[data_dir],
            num_work=num_work,
            tasks=paths_
        )
