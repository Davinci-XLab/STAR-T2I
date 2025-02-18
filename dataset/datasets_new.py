# common functions for training

import json
import traceback
import glob
import math
import os
import random
from io import BytesIO
import pathlib
import cv2

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode, transforms
import os,glob 
import numpy as np
from PIL import Image
import re
import pdb
import logging
import sys
sys.path.insert(0,'../')
# import cv2 
# print(cv2.__file__)

from dataset.lmdbdict import lmdbdict
# from dataset_parquet import create_parquet
from dataset.datasets_web import clean_mj_caption,countChinese,create_webdataset

logging.basicConfig(level=logging.INFO)

def containChinese(character):
    for cha in character:
        if '\u4e00' <= cha <= '\u9fa5':
            return True 
    return False 

def find_consecutive(string):
    pattern = r'(\w)\1{4,}'
    result = re.search(pattern, string)
    if result:
        return True
    else:
        return False 
    

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def get_transform(img_size):
    # mid_sz=math.floor(img_size*1.25)
    # mid_reso=(mid_sz,mid_sz)
    transform_list=[
        transforms.Resize((img_size,img_size), interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        # transforms.RandomCrop((img_size,img_size)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    return transforms.Compose(transform_list)

def load_mj_lmdb_from_dir(source_dir):
    all_info_json_files = glob.glob(os.path.join(source_dir,"MJData-*-info.json"))
    mj_lmdb_list = []
    # check main file and lock exists
    for info_json_file in all_info_json_files:
        main_file = info_json_file[:-10]
        lock_file = info_json_file[:-10]+"-lock"
        if os.path.exists(main_file) and os.path.exists(lock_file):
            mj_lmdb_list.append(main_file)
    # print(f"Found {len(mj_lmdb_list)} lmdbs in {source_dir}.")

    all_info_json_files = glob.glob(os.path.join(source_dir,"LAIONArtDataEn-*-info.json"))
    for info_json_file in all_info_json_files:
        main_file = info_json_file[:-10]
        lock_file = info_json_file[:-10]+"-lock"
        if os.path.exists(main_file) and os.path.exists(lock_file):
            mj_lmdb_list.append(main_file)

    all_info_json_files = glob.glob(os.path.join(source_dir,"tbfood60K-*-info.json"))
    for info_json_file in all_info_json_files:
        main_file = info_json_file[:-10]
        lock_file = info_json_file[:-10]+"-lock"
        if os.path.exists(main_file) and os.path.exists(lock_file):
            mj_lmdb_list.append(main_file)

    all_info_json_files = glob.glob(os.path.join(source_dir,"GenLow-*-info.json"))
    for info_json_file in all_info_json_files:
        main_file = info_json_file[:-10]
        lock_file = info_json_file[:-10]+"-lock"
        if os.path.exists(main_file) and os.path.exists(lock_file):
            mj_lmdb_list.append(main_file)
    
    all_info_json_files = glob.glob(os.path.join(source_dir,"freepicNotAI-*-info.json"))
    for info_json_file in all_info_json_files:
        main_file = info_json_file[:-10]
        lock_file = info_json_file[:-10]+"-lock"
        if os.path.exists(main_file) and os.path.exists(lock_file):
            mj_lmdb_list.append(main_file)
    
    all_info_json_files = glob.glob(os.path.join(source_dir,"freepicAI-*-info.json"))
    for info_json_file in all_info_json_files:
        main_file = info_json_file[:-10]
        lock_file = info_json_file[:-10]+"-lock"
        if os.path.exists(main_file) and os.path.exists(lock_file):
            mj_lmdb_list.append(main_file)

    for prefix in ['select lmdb tags']:
        all_info_json_files = glob.glob(os.path.join(source_dir, f"{prefix}-*-info.json"))
        for info_json_file in all_info_json_files:
            main_file = info_json_file[:-10]
            lock_file = info_json_file[:-10]+"-lock"
            if os.path.exists(main_file) and os.path.exists(lock_file):
                mj_lmdb_list.append(main_file)

    return mj_lmdb_list


def load_mj_lmdb_from_dirs(source_dirs):
    mj_lmdb_list = []
    for s in source_dirs:
        mj_lmdb_list.extend(load_mj_lmdb_from_dir(s))
    return mj_lmdb_list

def get_resize_height_width(img_h, img_w, bucket_h, bucket_w):
    new_h = bucket_h 
    new_w = int(new_h/img_h*img_w)
    if new_w < bucket_w:
        new_w_ = bucket_w
        new_h_ = int(new_w_/new_w*new_h)
        res = (new_w_,new_h_)
    else:
        res = (new_w,new_h)
    return res 


class MJDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        mj_lmdb_list,
        web_tar_list=None,
        parquet_list=None,
        v5_only=True,
        discard_list=[],
        caption_keys=["sharecaptioner_en"],
        human_threshold=999999,
        resolution=256,
        classifier_free_training_prob=0.1,
    ):
        '''
        mj_lmdb_list: lmdb主文件的路径列表
        mj_image_dir: 图像文件 mj_image_dir/20xx-xx-xx/jobid/xx.png
        size: 正方形尺寸为size x size，非正方形尺寸围绕size x size变化
        '''
        file_keys = set()
        self.dbs = []
        self.file_db_map = {}
        self.prompts = {}

        self.human_threshold = human_threshold
        self.resolution = resolution
        self.prompt_min_length=3
        self.prompt_max_length=1024
        self.classifier_free_training_prob = classifier_free_training_prob
        print("classifier-free training prob: ",classifier_free_training_prob)

        self.index2filekey = {}
        self.transform=get_transform(img_size=resolution)  # 不crop

        self.negative_tag=""
        self.len_effective_prompts=0
        cleaned_captions = {}
        
        resolution_discard = 0
        longBar_discard = 0
        prompt_discard = 0
        discard_count = 0
        read_portrait_info = 0
        update_captions = 0
        for mj_lmdb_path in mj_lmdb_list:
            # print(mj_lmdb_path)
            try:
                # print(" -- dataset reading ",mj_lmdb_path)
                db = lmdbdict(mj_lmdb_path,mode="r",map_size=1_073_741_824,max_readers=8)
                # print(len(db.keys()), mj_lmdb_path)
                with open(mj_lmdb_path+"-info.json") as fp:
                    info = json.load(fp)
                if os.path.exists(mj_lmdb_path+"-prompt.json"):
                    with open(mj_lmdb_path+"-prompt.json") as fp:
                        prompt_info = json.load(fp)
                else:
                    prompt_info = None

                db_ind = len(self.dbs)
                self.dbs.append(db)
                for file_key in db.keys():
                    if file_key in discard_list:
                        discard_count += 1
                        continue
                    j = info[file_key]

                    height, width = int(j['height']), int(j['width'])

                    # filter Long bar image
                    if max(height, width) / min(height, width) > 3:
                        longBar_discard += 1
                        continue 
                    # filter min resolution
                    if height < self.resolution or width < self.resolution:
                        resolution_discard += 1
                        continue 
                    # if height > self.min_resolution and width > self.min_resolution:
                    #     resolution_discard += 1
                    #     continue 

                    if prompt_info is None:#没有prompt.json的情况下，从info.json读
                        tmp_prompts = []
                        for caption_key in caption_keys:  # 有caption_keys的情况下，优先用caption_key
                            if caption_key in j:
                                tmp_prompts.append(j[caption_key])
                        if len(tmp_prompts)==0:#用caption_key读不到才用prompt作为key
                            tmp_prompts = [ j['prompt'] ]  # 有些数据集的prompt直接就是中文
                        # self.prompts[file_key] = [ j['prompt'] ]
                    else:
                        # 从prompt.json文件中读信息
                        if file_key in prompt_info:
                            tmp_prompts = []
                            for caption_key in caption_keys:
                                if caption_key in prompt_info[file_key]:
                                    tmp_prompts.append(prompt_info[file_key][caption_key])
                            if len(tmp_prompts)==0:#没有caption_keys的时候才用prompt作为key
                                if 'prompt' in prompt_info[file_key]:
                                    tmp_prompts = [prompt_info[file_key]['prompt']]
                        else:
                            tmp_prompts = []
                        
                    # print('tmp_prompt', len(tmp_prompts), file_key)
                    # 如果file_key在cleaned_captions中，那么仅采用cleaned_captions中的描述
                    # if remained_list is not None and file_key in cleaned_captions:
                    if file_key in cleaned_captions:
                        tmp_prompts = [cleaned_captions[file_key]]
                        update_captions += 1
                        # self.prompts[file_key] = [ prompt_info[file_key][caption_key] for caption_key in caption_keys ]
                    # three condition 1. containChinese 2.not too long or too short 3.not img2img
                    # tmp_prompts = [p for p in tmp_prompts if containChinese(p) and len(p) > self.prompt_min_length and len(p) <= self.prompt_max_length and 's.mj.run' not in p]
                    effective_prompts = []
                    for p in tmp_prompts:
                        if 's.mj.run' in p:
                            continue 
                        p = clean_mj_caption(p)
                        # 0.2 can be adjust
                        if len(p) > 0 and len(p) > self.prompt_min_length \
                            and len(p) <= self.prompt_max_length and not find_consecutive(p):
                            effective_prompts.append(p)

                    # print('effective prompt:', len(effective_prompts))
                    if len(effective_prompts) > 0:
                        self.prompts[file_key] = effective_prompts
                        self.len_effective_prompts+=len(effective_prompts)
                    else:
                        prompt_discard += 1
                        continue 

                    if 'command' in j and ("--v 5" not in j['command'] and "--version 5" not in j['command']) and v5_only:
                        continue 
                    if file_key in file_keys:
                        continue 

                    index = len(file_keys)
                    self.index2filekey[index] = file_key
                    file_keys.add(file_key)
                    self.file_db_map[file_key] = db_ind
                    
                    height, width = int(j['height']), int(j['width'])
            except:
                traceback.print_exc()
                print(mj_lmdb_path)
                continue
        self.num_instance_images = len(file_keys)
        print('==>prompt clean:', prompt_discard)#641356
        print('==>portrait info:', read_portrait_info)
        print('==>discard count:', discard_count)#3490159
        print('==>resolution_discard', resolution_discard)
        print('==>Long Bar discard', longBar_discard)
        print('==>update_captions', update_captions)
        print('==>effective images', self.num_instance_images)
        print('==>effective prompts', self.len_effective_prompts)
        self._length = self.num_instance_images

        # self.webdataset = None 
        # if web_tar_list is not None and len(web_tar_list) > 0:
        #     self.webdataset = create_webdataset(web_tar_list,
        #                             size=(self.resolution,self.resolution),
        #                             enable_text=True,
        #                             enable_image=True,
        #                             image_key="webp",
        #                             caption_key="txt",
        #                             total_length=100,
        #                             negative_tag=self.negative_tag,
        #                             prompt_length=[self.prompt_min_length,self.prompt_max_length],
        #                             classifier_free_training_prob=self.classifier_free_training_prob,
        #                             min_resolution=self.resolution,
        #                             cache_path=None)

            # webdataset_length = len(self.webdataset)
            # for i in range(webdataset_length):
            #     self.buckets['webdataset'].append(i+self.num_instance_images)
            # self.webdataset = iter(self.webdataset)
            # self._length = self.num_instance_images + webdataset_length
        # if parquet_list is not None and len(parquet_list)>0:
        #     # self.parquet_list = create_parquet()
        #     pass

        print('==>total effective images with web_tar', self._length)

    def get_image_from_lmdb(self,file_key):
        val = self.dbs[ self.file_db_map[file_key] ][file_key]
        with Image.open(BytesIO(val)) as img:
            return_img = img.copy()
        return return_img
        # return Image.open(BytesIO(val))

    def __len__(self):
        return self._length


    def __getitem__(self, index):
        # if index >= self.num_instance_images and self.webdataset:
        #     return next(self.webdataset) #TODO:webdataset不支持根据index筛选，这样搞是又问题的
        example = {}
        file_key = self.index2filekey[index]
        instance_image = self.get_image_from_lmdb(file_key)
       
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        img_w, img_h = instance_image.size   # 没错,w,h
        prompt = random.choice(self.prompts[file_key])

        # 为啥要用大于20的prompt做cfg，有依据吗
        if self.classifier_free_training_prob > 0.0 and len(prompt)>20 and random.random() < self.classifier_free_training_prob:
            prompt = self.negative_tag

        # 删除空格
        prompt = prompt.strip()
        # prompt = prompt.strip().replace(' ', '')
        example["prompt"] = prompt
        example["file_key"] = file_key

        # preprocessor = get_transform(img_size=img_size, bucket_size=(h, w), debug=self.debug, train_vae=self.train_vae)
        # center crop # 会导致四肢残缺
        img_size=min(img_h,img_w)
        crop_upx=(img_w-img_size)//2;crop_upy=(img_h-img_size)//2
        crop_downx=crop_upx+img_size;crop_downy=crop_upy+img_size
        instance_image = instance_image.crop((crop_upx,crop_upy,crop_downx,crop_downy))
        instance_image = self.transform(instance_image)

        # apply crop and resize:


        example["image"] = instance_image
        return example

if __name__=="__main__":
    batch_size = 2
    num_prepro_workers = 2
    preprocess = None 
    output_partition_count = 2
    actual_values = []
    # tokenizer = T5Tokenizer.from_pretrained("/home/data/aigc-models/IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese")
    lmdb_dirs=["/home/disk2/nfs/maxiaoxiao/datasets/lmdb_examples/laion_HD"]
    # web_tar_list_root=["/home/nfs/nfs-84/laion_HD/zhangtianyu/laion-hr-00118"]
    # web_tar_list = load_web_tar_from_dirs(web_tar_list_root)
    web_tar_list=['/home/nfs/nfs-84/laion_HD/zhangtianyu/laion-hr-00118/00000.tar']
    # ['__key__', '__url__', 'json', 'txt', 'webp']
    pdb.set_trace()
    # mj_lmdb_list = load_mj_lmdb_from_dirs(lmdb_dirs)
    mj_lmdb_list=[]
    print(len(mj_lmdb_list),len(web_tar_list))

    dataset = MJDataset(mj_lmdb_list,
            web_tar_list=web_tar_list,
            resolution=384,
            v5_only=False,
            caption_keys=["sharecaptioner_en"])
    print(len(dataset))
    print(dataset.num_instance_images)

    results=[]
    for i in range(100):
        item=dataset.__getitem__(i)
        results.append(item['file_key'])
    print(len(set(results)))

    # savedir='./test_data_oup'
    # pathlib.Path(savedir).mkdir(parents=True,exist_ok=True)
    pdb.set_trace()

    # train_dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=1,
    #         shuffle=False,
    #         num_workers=num_prepro_workers,
    #         # collate_fn=collate_fn,
    #         pin_memory=False,
    #         prefetch_factor=2,
    #     )
    # print('len:',len(train_dataloader))
    # for idx,item in enumerate(train_dataloader):
    #     print(item['image'].shape,item['prompt'],item["file_key"])
    #     # print(item['prompt'])
    #     Image.fromarray(((item['image'][0]+1)/2*255.).numpy().transpose(1,2,0).astype(np.uint8)).save(os.path.join(savedir,item['file_key'][0]+'.png'))
    # dataset = iter(dataset)