import sys
sys.path.insert(0,'..')
import os.path as osp
import webdataset as wds

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from dataset.datasets_new import MJDataset,load_mj_lmdb_from_dirs
from dataset.datasets_web import create_filter_function,create_preprocess_function,load_web_tar_from_dirs
import torch
import dist
import torch.distributed as tdist
import json
import pathlib
from PIL import Image
import numpy as np
import pdb
import os
import re


def gather_file_keys(file_keys):
    # 将 file_keys 转换为字符串形式的张量
    file_keys_tensor = torch.tensor([int(key) for key in file_keys], dtype=torch.int64).to(dist.get_device())

    # 创建用于收集结果的张量
    world_size = tdist.get_world_size()
    gathered_keys = [torch.empty_like(file_keys_tensor) for _ in range(world_size)]

    # 聚合 file_keys
    tdist.all_gather(gathered_keys, file_keys_tensor)

    # 在 global_rank=0 的进程上返回所有文件键
    if tdist.get_rank() == 0:
        all_file_keys = []
        for group in gathered_keys:
            all_file_keys.extend(group.cpu().numpy().astype(str).tolist())
        return all_file_keys

    return [0]  # 其他进程返回 None


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


# def build_dataset(
#     data_path: str, final_reso: int,
#     hflip=False, mid_reso=1.125,
# ):
#     # build augmentations
#     mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
#     train_aug, val_aug = [
#         transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
#         transforms.RandomCrop((final_reso, final_reso)),
#         transforms.ToTensor(), normalize_01_into_pm1,
#     ], [
#         transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
#         transforms.CenterCrop((final_reso, final_reso)),
#         transforms.ToTensor(), normalize_01_into_pm1,
#     ]
#     if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
#     train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
#     # build dataset
#     train_set = DatasetFolder(root=osp.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=train_aug)
#     val_set = DatasetFolder(root=osp.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=val_aug)
#     num_classes = 1000
#     print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
#     print_aug(train_aug, '[train]')
#     print_aug(val_aug, '[val]')
    
#     return num_classes, train_set, val_set

def get_low_aesthetic_list(save_root='/nfs-25/liangtao/improved-aesthetic-predictor-main/MJ_aesthetic_scores', aesthetic_thre=5):
    assert os.path.exists(save_root)
    low_aesthetic_list = []
    for name in os.listdir(save_root):
        json_path = os.path.join(save_root, name)
        with open(json_path) as fp:
            data = json.load(fp)
        for key, value in data.items():
            if isinstance(value, dict):
                if value['aesthetic_score'] <= aesthetic_thre:
                    low_aesthetic_list.append(key)
            else:
                if value <= aesthetic_thre:
                    low_aesthetic_list.append(key)
    return low_aesthetic_list


def get_low_aesthetic_freepic(save_root='/nfs-25/liangtao/improved-aesthetic-predictor-main/freepic_aesthetic_scores', aesthetic_thre=5.0):  # 5.5
    assert os.path.exists(save_root)
    low_aesthetic_list = []
    if not os.path.exists(save_root):
        return low_aesthetic_list
    for name in os.listdir(save_root):
        if 'freepic_ai' in name:
            prefix = 'freepicAI-'
        else:
            prefix = 'freepicNotAI-'
        json_path = os.path.join(save_root, name)
        with open(json_path) as fp:
            data = json.load(fp)
        for key, value in data.items():
            if isinstance(value, dict):
                if value['aesthetic_score'] <= aesthetic_thre:
                    low_aesthetic_list.append(prefix+key)
            else:
                if value <= aesthetic_thre:
                    low_aesthetic_list.append(prefix+key)
    return low_aesthetic_list



# 之前一直设置的-1，4台机器训练就换成0，也可以直接做成-0.5
def get_low_quality_list(save_root='/nfs-25/liangtao/ImageReward/ImageReward-main/MJ_reward_scores', reward_thre=-1):
    print('using reward thre=',reward_thre)
    assert os.path.exists(save_root)
    low_quality_list = []
    for name in os.listdir(save_root):
        json_path = os.path.join(save_root, name)
        with open(json_path) as fp:
            data = json.load(fp)
        for key, value in data.items():
            if isinstance(value, dict):
                if value['reward_score'] <= reward_thre:
                    low_quality_list.append(key)
            else:
                if value <= reward_thre:
                    low_quality_list.append(key)
    return low_quality_list


def get_collage_list(save_root='/nfs-25/liangtao/ImageDatasetCleaner/collageInfo'):
    assert os.path.exists(save_root)
    collage_list = []
    for name in os.listdir(save_root):
        json_path = os.path.join(save_root, name)
        with open(json_path) as fp:
            data = json.load(fp)
        for key, value in data.items():
            collage_list.append(key)
    return collage_list

def get_remained_list(save_root='/nfs-25/liangtao/ImageReward/ImageReward-main/remained'):
    assert os.path.exists(save_root)
    remained_list = []
    for name in os.listdir(save_root):
        json_path = os.path.join(save_root, name)
        with open(json_path) as fp:
            data = json.load(fp)
        for key, value in data.items():
            if 'unsplash' in key:
                for prefix in ['unsplash25K-001-', 'unsplash25K002-', 'unsplash25K003-', 'unsplash25K004-', 'unsplash25K005-']:
                    remained_list.append(prefix+key)
            else:
                remained_list.append(key)
    return remained_list


def build_dataset(args):
    data_json=args.data_args
    lmdb_dirs=data_json['lmdb_dirs']
    mj_lmdb_list = load_mj_lmdb_from_dirs(lmdb_dirs)
    print('length of trainset ----- ',len(mj_lmdb_list))
    lmdb_dirs_val=data_json['lmdb_dirs_val']
    mj_lmdb_list_val = load_mj_lmdb_from_dirs(lmdb_dirs_val)
    print('length of valset ----- ',len(mj_lmdb_list_val))

    low_aesthetic_list = get_low_aesthetic_list()
    print('==>low aesthetic_list', len(low_aesthetic_list))#291886
    collage_list = get_collage_list()
    print('==>collage_list', len(collage_list))#139656
    low_aesthetic_freepic = get_low_aesthetic_freepic()
    print('==>low aesthetic_freepic', len(low_aesthetic_freepic))#114401
    low_quality_list = get_low_quality_list(reward_thre=args.reward_thre)
    # low_quality_list = []
    print('==>low quality_list', len(low_quality_list))#2165981

    low_quality_list.extend(low_aesthetic_freepic)  # extend freepic default
    low_quality_list.extend(collage_list)  # extend collage list default

    discard_list= []
    if args.enable_discard:
        for file_name in ['/nfs-25/liangtao/to_others/to_liangtao/mj_discard.json', 
                        '/nfs-25/liangtao/to_others/to_liangtao/mj_discard_v2.json']:
            assert os.path.exists(file_name)
            with open(file_name) as fp:
                data = json.load(fp)
            discard_list.extend(list(data.keys()))

    if args.enable_discard_low:
        discard_list.extend(low_quality_list)
        low_quality_list = None

    if args.enable_discard_aesthetic:
        discard_list.extend(low_aesthetic_list)
        low_quality_list = None

    train_set = MJDataset(mj_lmdb_list,
                        resolution=args.data_load_reso,
                        discard_list=set(discard_list),
                        v5_only=False,
                        caption_keys=["prompt", 'florence_en', "gpt_caption", "minicpm_caption"],
                        classifier_free_training_prob=args.classifier_free_training_prob)
    val_set = MJDataset(mj_lmdb_list_val,
                        resolution=args.data_load_reso,
                        discard_list=set(discard_list),
                        v5_only=False,
                        caption_keys=["prompt", 'florence_en', "gpt_caption", "minicpm_caption"],
                        #caption_keys=["sharecaptioner_en"],
                        classifier_free_training_prob=args.classifier_free_training_prob)
    return train_set,val_set


def filter_dataset(item):
    return create_filter_function(
        enable_text=True,
        enable_image=False,  # for debug
        image_key=["webp", "jpg", "png", "jpeg"],
        prompt_length=[3,2000],
        min_resolution=256,
        bucket_ratio=None
    )(item)

def preprocess_dataset(item):
    return create_preprocess_function(
        size=(256, 256),
        enable_text=True,
        enable_image=True,
        image_key=["webp", "jpg", "png", "jpeg"],
        negative_tag="",
        classifier_free_training_prob=0.1
    )(item)

def build_dataset_webtar(args):
    web_tar_list = load_web_tar_from_dirs(args.web_tar_root)
    web_tar_list_val = load_web_tar_from_dirs(args.web_tar_root_val)
    print('length of train webtar ----- ', len(web_tar_list))
    print('length of val webtar ----- ', len(web_tar_list_val))

    train_set = wds.DataPipeline(
        wds.SimpleShardList(web_tar_list),
        wds.tarfile_to_samples(),
        wds.split_by_node,
        wds.split_by_worker,
        wds.select(filter_dataset),
        wds.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    )#.with_length(args.web_tar_len)

    val_set = wds.DataPipeline(
        wds.SimpleShardList(web_tar_list_val),
        wds.tarfile_to_samples(),
        wds.select(filter_dataset),
        wds.shuffle(args.web_tar_len_val),
        wds.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    ).with_length(args.web_tar_len_val)

    return train_set, val_set




def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')


if __name__=="__main__":
    import os
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import webdataset as wds
    from utils.utils import format_sentence
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    class Args:
        def __init__(self):
            pass
    args=Args()
    args.data_args={
        "lmdb_dirs": [
            # "/home/nfs/nfs-141/maxiaoxiao/datasets/dpg_sdv3_packs",
            # "/home/nfs/nfs-141/maxiaoxiao/datasets/geneval_flux_packs",
            # "/home/nfs/nfs-141/maxiaoxiao/datasets/geneval_sdv3_packs",
            # "/home/nfs/nfs-141/maxiaoxiao/datasets/my_vqa_data_batch2_packs",
            "/home/nfs/nfs-141/maxiaoxiao/datasets/MJHQ30K_packs",
        ],
        "lmdb_dirs_val": [
            "/nfs-30/atlas-pipeline-train-data/MJPacks-F8"
        ]
    }
    args.reward_thre=0
    args.data_load_reso=1024
    args.enable_discard=True
    args.prompt_length=[3,2000]
    args.classifier_free_training_prob=0.0
    args.web_tar_len=100
    args.web_tar_len_val=100
    args.glb_batch_size=10

    data_json=args.data_args
    lmdb_dirs=data_json['lmdb_dirs']
    mj_lmdb_list = load_mj_lmdb_from_dirs(lmdb_dirs)
    print('length of trainset ----- ',len(mj_lmdb_list))
    train_set = MJDataset(mj_lmdb_list,
                        resolution=args.data_load_reso,
                        discard_list=set([]),
                        v5_only=False,
                        caption_keys=["sharecaptioner_en","en_prompt"],
                        classifier_free_training_prob=args.classifier_free_training_prob)

    for item in train_set:
        print(item['prompt'])
        if not os.path.exists('./oup_imgs_geneval'):os.makedirs('./oup_imgs_geneval')
        transforms.ToPILImage()((item['image']+ 1) / 2).save('./oup_imgs_geneval/%s.png'%format_sentence(item['prompt']+'xx'))
        pdb.set_trace()


    # 下为测试webtar dataset的代码
    '''
    def cleanup():
        dist.destroy_process_group()

    class Args:
        def __init__(self):
            pass
    args=Args()
    args.data_load_reso=512
    args.prompt_length=[3,2000]
    args.classifier_free_training_prob=0.1
    args.web_tar_len=100
    args.web_tar_len_val=100
    args.glb_batch_size=10
    args.web_tar_root=[
                        # "/nfs-80/laion_HD/zhoumohan/part-00075",
                        # "/nfs-80/laion_HD/zhoumohan/part-00122",
                        # "/nfs-80/laion_HD/zhoumohan/part-00123",
                        # "/nfs-77/laion_HD/zhoumohan/part-00010",
                        # "/nfs-77/laion_HD/zhoumohan/part-00011",
                        # "/nfs-77/laion_HD/zhoumohan/part-00012",
                        # "/nfs-77/laion_HD/zhoumohan/part-00013",
                        # "/nfs-77/laion_HD/zhoumohan/part-00014",
                        # "/nfs-78/laion_HD/zhoumohan/part-00015",
                        # "/nfs-78/laion_HD/zhoumohan/part-00016",#6M左右
                        # "/nfs-78/laion_HD/zhoumohan/part-00017",
                        # "/nfs-78/laion_HD/zhoumohan/part-00018",
                        # "/nfs-78/laion_HD/zhoumohan/part-00019",
                        # "/nfs-82/laion_HD/zhangtianyu/laion-hr-00055",
                        # "/nfs-82/laion_HD/zhangtianyu/laion-hr-00056",
                        # "/nfs-82/laion_HD/zhangtianyu/laion-hr-00057",
                        # "/nfs-82/laion_HD/zhangtianyu/laion-hr-00058",
                        # "/nfs-82/laion_HD/zhangtianyu/laion-hr-00059",
                        # "/nfs-79/laion_HD/zhoumohan/part-00070",
                        # "/nfs-79/laion_HD/zhoumohan/part-00071",
                        # "/nfs-79/laion_HD/zhoumohan/part-00072",
                        # "/nfs-79/laion_HD/zhoumohan/part-00073",
                        # "/nfs-79/laion_HD/zhoumohan/part-00074",
                        # "/nfs-83/laion_HD/zhangtianyu/laion-hr-00110",
                        # "/nfs-83/laion_HD/zhangtianyu/laion-hr-00111",
                        # "/nfs-83/laion_HD/zhangtianyu/laion-hr-00112",
                        # "/nfs-83/laion_HD/zhangtianyu/laion-hr-00113",
                        # "/nfs-83/laion_HD/zhangtianyu/laion-hr-00114",
                        # "/nfs-84/laion_HD/zhangtianyu/laion-hr-00115",
                        # "/nfs-84/laion_HD/zhangtianyu/laion-hr-00116",
                        # "/nfs-84/laion_HD/zhangtianyu/laion-hr-00117",
                        # "/nfs-84/laion_HD/zhangtianyu/laion-hr-00118",
                        # "/nfs-84/laion_HD/zhangtianyu/laion-hr-00119",#15M左右
                        "/nfs-43/laion_HD/jingyifei/laion-high-resolution-output_part-00082-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
                        "/nfs-43/laion_HD/jingyifei/laion-high-resolution-output_part-00083-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
                        "/nfs-43/laion_HD/jingyifei/laion-high-resolution-output_part-00084-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
                        "/nfs-43/laion_HD/jingyifei/laion-high-resolution-output_part-00085-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
                        "/nfs-43/laion_HD/jingyifei/laion-high-resolution-output_part-00086-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
                        "/nfs-44/laion_HD/jingyifei/laion-high-resolution-output_part-00026-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
                        "/nfs-44/laion_HD/jingyifei/laion-high-resolution-output_part-00027-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
                        "/nfs-44/laion_HD/jingyifei/laion-high-resolution-output_part-00028-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
                        "/nfs-44/laion_HD/jingyifei/laion-high-resolution-output_part-00029-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
                        "/nfs-44/laion_HD/jingyifei/laion-high-resolution-output_part-00080-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
                        "/nfs-45/laion_HD/jingyifei/laion-high-resolution-output_part-00020-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
                        "/nfs-45/laion_HD/jingyifei/laion-high-resolution-output_part-00021-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
                        "/nfs-45/laion_HD/jingyifei/laion-high-resolution-output_part-00022-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
                        "/nfs-45/laion_HD/jingyifei/laion-high-resolution-output_part-00023-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
                        "/nfs-45/laion_HD/jingyifei/laion-high-resolution-output_part-00024-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet"#8M+15M=23M
                        ]
    args.web_tar_root_val=["/nfs-78/laion_HD/zhoumohan/part-00019"]

    def custom_collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None  # Handle empty batch case
        return torch.utils.data.dataloader.default_collate(batch)

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)  # 设置当前设备为本地进程设备

    trainset,_=build_dataset_webtar(args)
    # sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    loader=DataLoader(trainset,batch_size=8,num_workers=4,shuffle=False,collate_fn=custom_collate_fn)
    # loader = wds.WebLoader(
    #     trainset,
    #     batch_size=4,
    #     shuffle=False,
    #     num_workers=4,
    #     persistent_workers=4 > 0,
    # )
    # iters_train = len(loader)
    # print('train loader length:',iters_train)
    # pdb.set_trace()
    # savedir='./test_data_oup'
    # pathlib.Path(savedir).mkdir(parents=True,exist_ok=True)

    # for idx,item in enumerate(loader):
    #     pdb.set_trace()
    #     print(item['prompt'],item['file_key'],item['image'].shape,item['image'].max(),item['image'].min())
    #     Image.fromarray(((item['image'][0]+1)/2*255.).numpy().transpose(1,2,0).astype(np.uint8)).save(os.path.join(savedir,item['file_key'][0]+'.png'))


    total_file_keys=[]
    for epoch in range(1):
        file_keys=[]
        for idx,item in enumerate(loader):
            # print(item['prompt'],item['file_key'],'node=',dist.get_world_size())
            file_keys.extend(item['file_key'])  # 假设数据通过某种方式包装成 batch
            if idx%500==0:
                print(f"Rank {rank}: Prompt={item['prompt']}, File Key={item['file_key']}, url={item['url']}")
                print(f"Currently rank {rank} collected file keys: {len(set(file_keys)),len(file_keys)}\n")
        total_file_keys.extend(file_keys)
        print(f"Rank {rank} collected file keys: {len(set(file_keys)),len(file_keys)}")
    print(f"Rank {rank} collected total file keys: {len(set(total_file_keys)),len(total_file_keys)}")
    '''