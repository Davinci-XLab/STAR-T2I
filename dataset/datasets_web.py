import os, io, glob 
from PIL import Image 
import webdataset as wds  # pylint: disable=import-outside-toplevel
import pdb
import json 
import torch 
import random 
import sys 
# sys.path.insert(0, '/usr/local/lib/python3.8/dist-packages/opencv-contrib-python')
import cv2 
print(cv2.__file__)
import numpy as np 
from torchvision import transforms
import dist
import json
import re 
os.environ["WDS_VERBOSE_CACHE"] = "1"


# 在去掉url后使用
def countChinese(character):
    '''是否包含中文字符'''
    count = 0
    for cha in character:
        if '\u4e00' <= cha <= '\u9fa5':
            count += 1
    return count / len(character.split())


def countEnglish(character):
    '''Calculate the proportion of English alphabetic characters in a string'''
    count = 0
    total_characters = len(character.replace(" ", ""))  # Remove spaces to count only characters
    for cha in character:
        if ('a' <= cha <= 'z') or ('A' <= cha <= 'Z'):
            count += 1
    return count / total_characters if total_characters > 0 else 0


def clean_mj_caption(caption):
    # url
    caption = re.sub(r"file:///.+\.(jpg|png|JPG)", '', caption)  # 删除以file开头，以png或者jpg结束的超链接，top
    caption = re.sub(r'https?[^ ]*\.(?:png|jpg)\b', '', caption)  # 删除以https/http开头，以png或者jpg结束的超链接，top

    caption = re.sub(
        r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it|png|jpg)[\w/-]*\b\/?(?!@)))',  # noqa
        '', caption)  # regex for urls
    caption = re.sub(
        r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it|png|jpg)[\w/-]*\b\/?(?!@)))',  # noqa
        '', caption)  # regex for urls

    caption = re.sub(r"--(v|version)\s+\d+(\.\d+)*", '', caption)
    caption = re.sub(r"--(v|version|ar|iw)\s+\S+", '', caption)
    caption = re.sub(r"blob:", '', caption)  # blob:
    caption = re.sub(r"blob：", '', caption)  # blob:
    caption = re.sub(r'^\/(image|imagine)\b', '', caption)
    caption = re.sub(r"@[^@]+@", '', caption)
    caption = re.sub(r"^[^\w\s]+", "", caption)
    return caption.strip()

def get_low_aesthetic_laionHD(save_root='/home/work/improved-aesthetic-predictor-main/laionHD_aesthetic_scores', aesthetic_thre=5.0):  # 5.5
    low_aesthetic_list = []
    if not os.path.exists(save_root):
        return low_aesthetic_list
    for name in os.listdir(save_root):
        json_path = os.path.join(save_root, name)
        with open(json_path) as fp:
            data = json.load(fp)
        for key, value in data.items():
            if isinstance(value, dict):
                if value['aesthetic_score'] <= aesthetic_thre:
                    low_aesthetic_list.append(key)
                else:
                    width = value['width']
                    height = value['height']
                    if max(height, width) / min(height, width) > 2.5 or min(height, width) < 768:
                        low_aesthetic_list.append(key)
            else:
                if value <= aesthetic_thre:
                    low_aesthetic_list.append(key)
    return set(low_aesthetic_list)

def get_llava_infos(save_root='/home/storages/dev95/disk4/libai/dev/image-caption-demo/outputs/'):  # must /
    discard_set = get_low_aesthetic_laionHD()
    prompt_infos = {}
    for part in ['laion_pop_highres']:
        all_jsons = glob.glob(save_root + part + '/laion_pop*.json')
        print('==> laion_pop:', len(all_jsons))
        for json_file in all_jsons:
            with open(json_file) as fp:
                for line in fp.readlines():
                    data = json.loads(line.strip())
                    assert 'image_name' in data
                    assert 'llava_caption' in data
                    image_name = data['image_name']
                    image_name = image_name.split('.')[0]
                    # if image_name not in discard_set:
                    #     prompt_infos[image_name] = {'llava':data['llava_caption']}
                    prompt_infos[image_name] = {'llava':data['llava_caption']}
    ''' 
    for part in ['sam_part1_json', 'sam_part4_json', 'sam_part6_json']:
        all_jsons = glob.glob(save_root + part + '/sa_*.json')
        print('==> SAM:', len(all_jsons))
        for json_file in all_jsons:
            with open(json_file) as fp:
                for line in fp.readlines():
                    data = json.loads(line.strip())
                    assert 'image_name' in data
                    assert 'llava_caption' in data
                    image_name = data['image_name'][2:] if data['image_name'].startswith('./') else data['image_name']
                    image_name = image_name.split('.')[0]
                    if image_name not in discard_set:
                        prompt_infos[image_name] = {'llava':data['llava_caption']}
    '''
    print('==> loading llava prompts:', len(prompt_infos))
    return prompt_infos
prompt_infos = get_llava_infos()

def get_resize_height_width(img_h, img_w, bucket_h, bucket_w):
    new_h = bucket_h 
    new_w = int(new_h/img_h*img_w)
    if new_w < bucket_w:
        new_w_ = bucket_w
        new_h_ = int(new_w_/new_w*new_h)
        res = (new_h_, new_w_)
    else:
        res = (new_h, new_w)
    return res 

def load_web_tar_from_dir(web_dir):
    all_web_tars = glob.glob(os.path.join(web_dir, '*.tar'))
    all_web_tars_list = []
    for web_tar in all_web_tars:
        if os.path.exists(web_tar):
            all_web_tars_list.append(web_tar)
    return all_web_tars_list 

def load_web_tar_from_dirs(source_dirs):
    all_web_tars = []
    corrupted_tar_list=[
                        '/nfs-80/laion_HD/zhoumohan/part-00075/00019.tar',
                        '/nfs-80/laion_HD/zhoumohan/part-00075/00096.tar',
                        '/nfs-80/laion_HD/zhoumohan/part-00075/00097.tar',
                        '/nfs-80/laion_HD/zhoumohan/part-00075/00098.tar',
                        '/nfs-80/laion_HD/zhoumohan/part-00075/00099.tar',
                        '/nfs-80/laion_HD/zhoumohan/part-00123/00111.tar',
                        '/nfs-80/laion_HD/zhoumohan/part-00123/00065.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00016/00040.tar',
                        '/nfs-79/laion_HD/zhoumohan/part-00070/00125.tar',
                        '/nfs-79/laion_HD/zhoumohan/part-00074/00060.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00121.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00117.tar',
                        '/nfs-79/laion_HD/zhoumohan/part-00074/00058.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00134.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00136.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00122.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00135.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00133.tar',
                        '/nfs-79/laion_HD/zhoumohan/part-00070/00124.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00116.tar',
                        '/nfs-79/laion_HD/zhoumohan/part-00074/00057.tar',
                        '/nfs-79/laion_HD/zhoumohan/part-00074/00059.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00018/00049.tar',
                        '/nfs-79/laion_HD/zhoumohan/part-00070/00127.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00018/00048.tar',
                        '/nfs-77/laion_HD/zhoumohan/part-00012/00124.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00120.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00130.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00127.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00131.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00125.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00018/00050.tar',
                        '/nfs-79/laion_HD/zhoumohan/part-00070/00126.tar',
                        '/nfs-79/laion_HD/zhoumohan/part-00070/00126.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00118.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00129.tar',
                        '/nfs-77/laion_HD/zhoumohan/part-00011/00047.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00018/00051.tar',
                        '/nfs-79/laion_HD/zhoumohan/part-00073/00088.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00124.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00132.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00126.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00119.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00128.tar',
                        '/nfs-78/laion_HD/zhoumohan/part-00017/00123.tar'
                        ]
    for s in source_dirs:
        all_web_tars.extend(load_web_tar_from_dir(s))
    return [tar for tar in all_web_tars if tar not in corrupted_tar_list]

def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src

class Preprocessor(object):
    def __init__(self, img_size, bucket_size):
        img_h, img_w = img_size
        self.height, self.width = bucket_size
        self.resized_h = int(img_h * 1.05)
        self.resized_w = int(img_w * 1.05)

        self.x = np.random.randint(low=0, high=self.resized_w - self.width + 1, size=1)[0]
        self.y = np.random.randint(low=0, high=self.resized_h - self.height + 1, size=1)[0]
        self.flip = False 

    def __call__(self, inputs, convert_to_numpy, normalize):
        if convert_to_numpy:  # 转为 numpy 格式
            inputs = inputs.resize((self.resized_w, self.resized_h), Image.ANTIALIAS)  # 抗锯齿
            inputs = np.array(inputs)
        else:
            inputs = cv2.resize(inputs, (self.resized_w, self.resized_h), interpolation=cv2.INTER_LINEAR)
        inputs = inputs[self.y:self.y + self.height, self.x: self.x + self.width, ...]  # 随机裁剪

        if inputs.ndim == 2:
            inputs = np.expand_dims(inputs, axis=-1)
        if self.flip:
            inputs = np.fliplr(inputs)
            inputs = np.ascontiguousarray(inputs)

        if normalize:
            inputs = transforms.ToTensor()(inputs)
            inputs = transforms.Normalize([0.5], [0.5])(inputs)
        else:
            inputs = inputs.transpose(2, 0, 1)
            inputs = torch.from_numpy(inputs)
        return inputs

def create_preprocess_function(
            size=(768, 768),
            enable_text=True,
            enable_image=True,
            image_key=["webp","jpg","png","jpeg"],
            negative_tag="",
            classifier_free_training_prob=0.0,
):
    def preprocess_dataset(item):
        output = {}
        online_tag = None
        # print(item['txt'],item['json'],item['__key__'],item['__url__'])
        if enable_image:
            efficient_key=next((key for key in image_key if key in item), None)
            image_data = item[efficient_key]
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            img_w, img_h = image.size 
            img_size = get_resize_height_width(img_h=img_h, img_w=img_w, bucket_h=size[0], bucket_w=size[1])
            image_tensor = get_transform(img_size=img_size, bucket_size=size, debug=False)(image)
            # output["image_filename"] = item["__key__"]
            output["image"] = image_tensor 
            output["target_size"] = size
            # print(size)

        if enable_text:
            # 如何获取到key
            file_key = item['__key__']
            # if file_key in prompt_infos:
            #     if random.random() > 0.5:
            #         prompt = prompt_infos[file_key]['llava'].strip()[7:]  # 这幅图片展示了
            #     else:
            #         prompt = prompt_infos[file_key]['llava'].strip()
            # else:
            json_data = json.loads(item['json'].decode("utf-8"))
            prompt=json_data.get('caption',None)
                # caption = text.decode("utf-8")
            prompt = prompt.strip()
            prompt = clean_mj_caption(prompt)
            # enable  classifier training
            if classifier_free_training_prob > 0.0 and random.random() < classifier_free_training_prob:
                prompt = negative_tag
            
            output["file_key"] = file_key
            output["prompt"] = prompt
            output["url"]=item['__url__']
            # print(file_key,prompt) 

        return output
    return preprocess_dataset

def create_filter_function(
        enable_text=True,
        enable_image=True,
        image_key=["webp","jpg","png","jpeg"],
        prompt_length=[0,300],
        min_resolution=768,
        bucket_ratio=None,
        language_discard=['zn','fa','ru','ja','el']
):
    def filter_dataset(item):
        decision = 0
        # pdb.set_trace()
        # print(item['txt'],item['json'],item['__key__'],item['__url__'])
        if enable_image and (not True in [key in item.keys() for key in image_key]):
            decision = 1
        # if enable_image and image_key in item:
        #     image_data = item[image_key]
        #     image = Image.open(io.BytesIO(image_data))
        #     if len(image.getbands()) != 3:
        #         return False 

        # if enable_metadata and "json" not in item:
        #     return False
        if 'json' in item:
            infos = json.loads(item['json'])
            if enable_text and item['__key__'] not in prompt_infos:#没有llava recap的项
                prompt_cur=infos.get('caption',None)#有一些奇奇怪怪国家的语言，countchinese改成countenglish
                if infos['LANGUAGE'] not in language_discard \
                and countEnglish(prompt_cur)>0.6 \
                    and len(prompt_cur)>prompt_length[0] \
                        and len(prompt_cur)<prompt_length[1]:
                    decision = 0
                else:
                    decision = 2
            if 'pwatermark' in infos and infos['pwatermark'] is not None and infos['pwatermark'] > 0.6:
                decision = 3 
            if 'punsafe' in infos and infos['punsafe'] is not None and infos['punsafe'] > 0.5:
                decision = 4
            
            if 'width' in infos and 'height' in infos and infos['height'] and infos['width'] and max(infos['height'], infos['width']) / min(infos['height'], infos['width']) > 3.0:
                decision = 5 
            
            if 'width' in infos and 'height' in infos and infos['height'] and infos['width'] and bucket_ratio is not None:
                rs = infos['height']/infos['width']
                if rs <= bucket_ratio[0] or rs > bucket_ratio[1]:
                    decision = 6 
                
            if 'width' in infos and infos['width'] and infos['width'] < min_resolution:
                decision = 7 
            if 'height' in infos and infos['height'] and infos['height'] < min_resolution:
                decision = 8
            # if infos['similarity']<0.25:
            #     print(infos['similarity'],infos['url'],prompt_cur)
        else:
            decision = 9 
        return True if decision==0 else False
    return filter_dataset


def create_webdataset(
    urls,
    size=(768, 768),
    enable_text=True,
    enable_image=True,
    image_key="jpg",
    caption_key="txt",
    total_length=1000000,
    positive_tag=False,
    negative_tag="",
    prompt_length=[0,300],
    classifier_free_training_prob=0.0,
    min_resolution=768,
    cache_path=None,
    bucket_ratio=None,
):
    """Create a WebDataset reader, it can read a webdataset of image, text and json"""
    # select_files=['zh_google','txt', 'json', 'jpg']
    def my_split_by_worker(urls):
        wi = torch.utils.data.get_worker_info()
        if wi is None:
            return urls
        else:
            return urls[wi.id::wi.num_workers]
    def my_split_by_node(urls):
        node_id, node_count = torch.distributed.get_rank(), torch.distributed.get_world_size()
        return urls[node_id::node_count]
    
    # dataset = wds.ShardList(urls, splitter=wds.split_by_worker, nodesplitter=wds.split_by_node, shuffle=False)
    dataset = wds.WebDataset(urls, 
                             cache_dir=cache_path, 
                             cache_size=10 ** 10, 
                             handler=wds.handlers.warn_and_continue, 
                             shardshuffle=True,
                             resampled=True,
                             nodesplitter=nodesplitter)#这这个nodesplitter有必要吗？

    def filter_dataset(item):
        # print('preprocess begin...')
        # pdb.set_trace()
        decision = 0
        if enable_image and image_key not in item:
            decision = 1
        # if enable_image and image_key in item:
        #     image_data = item[image_key]
        #     image = Image.open(io.BytesIO(image_data))
        #     if len(image.getbands()) != 3:
        #         return False 

        # if enable_metadata and "json" not in item:
        #     return False
        if 'json' in item:
            infos = json.loads(item['json'])
            if enable_text and item['__key__'] not in prompt_infos:#没有llava recap的项
                prompt_cur=infos.get('caption',None)#有一些奇奇怪怪国家的语言，countchinese改成countenglish
                if infos['LANGUAGE']=='en' and len(prompt_cur)>prompt_length[0] and len(prompt_cur)<prompt_length[1]:
                    decision = 0
                else:
                    decision = 2
            if 'pwatermark' in infos and infos['pwatermark'] is not None and infos['pwatermark'] > 0.6:
                decision = 3 
            if 'punsafe' in infos and infos['punsafe'] is not None and infos['punsafe'] > 0.5:
                decision = 4
            
            if 'width' in infos and 'height' in infos and infos['height'] and infos['width'] and max(infos['height'], infos['width']) / min(infos['height'], infos['width']) > 3.0:
                decision = 5 
            
            if 'width' in infos and 'height' in infos and infos['height'] and infos['width'] and bucket_ratio is not None:
                rs = infos['height']/infos['width']
                if rs <= bucket_ratio[0] or rs > bucket_ratio[1]:
                    decision = 6 
                
            if 'width' in infos and infos['width'] and infos['width'] < min_resolution:
                decision = 7 
            if 'height' in infos and infos['height'] and infos['height'] < min_resolution:
                decision = 8
            # if infos['similarity']<0.25:
            #     print(infos['similarity'],infos['url'],prompt_cur)
        else:
            decision = 9 
        return True if decision==0 else False
    # with_epoch()可以直接用于DDP，不需要nodesplitter
    # filtered_dataset = dataset.select(filter_dataset).with_length(total_length).shuffle(total_length)
    filtered_dataset = dataset.select(filter_dataset).shard(dist.get_world_size(),dist.get_rank())
    # pdb.set_trace()

    def preprocess_dataset(item):
        output = {}
        online_tag = None
        if enable_image:
            image_data = item[image_key]
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            img_w, img_h = image.size 
            img_size = get_resize_height_width(img_h=img_h, img_w=img_w, bucket_h=size[0], bucket_w=size[1])
            image_tensor = get_transform(img_size=img_size, bucket_size=size, debug=False)(image)
            # output["image_filename"] = item["__key__"]
            output["image"] = image_tensor 
            output["target_size"] = size
            # print(size)

        if enable_text:
            # 如何获取到key
            file_key = item['__key__']
            # if file_key in prompt_infos:
            #     if random.random() > 0.5:
            #         prompt = prompt_infos[file_key]['llava'].strip()[7:]  # 这幅图片展示了
            #     else:
            #         prompt = prompt_infos[file_key]['llava'].strip()
            # else:
            json_data = json.loads(item['json'].decode("utf-8"))
            prompt=json_data.get('caption',None)
                # caption = text.decode("utf-8")
            prompt = prompt.strip()
            prompt = clean_mj_caption(prompt)
            # enable  classifier training
            if classifier_free_training_prob > 0.0 and random.random() < classifier_free_training_prob:
                prompt = negative_tag
            
            output["file_key"] = file_key
            output["prompt"] = prompt
            output["url"]=item['__url__']
            # print(file_key,prompt) 

        return output

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    return transformed_dataset

# img_size按照512x512去设置等比例缩放
def get_transform(img_size, bucket_size=(512, 512), debug=False):
    img_h, img_w = img_size
    height, width = bucket_size
    transform_list = [
                transforms.Resize((int(img_h*1.05),int(img_w*1.05)), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop((height, width)),
                # transforms.RandomCrop((height, width)),
                # transforms.RandomHorizontalFlip()
        ]
    if not debug:
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    return transforms.Compose(transform_list)


def collate_fn(examples):
    for e in examples:
        print(e['prompt'])
    pixel_values = torch.stack([example["image"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

if __name__ == '__main__':
    from transformers import XLMRobertaTokenizer
    batch_size = 2
    num_prepro_workers = 2
    preprocess = None 
    output_partition_count = 2
    actual_values = []
    web_tar_list_root=["/home/nfs/nfs-84/laion_HD/zhangtianyu/laion-hr-00118"]
    # web_tar_list = load_web_tar_from_dirs(web_tar_list_root)
    web_tar_list=['/home/nfs/nfs-84/laion_HD/zhangtianyu/laion-hr-00118/00000.tar']
    dataset = wds.WebDataset(web_tar_list, 
                             cache_dir=None, 
                             cache_size=10 ** 10, 
                             handler=wds.handlers.warn_and_continue, 
                             shardshuffle=True,
                             resampled=True,
                             nodesplitter=nodesplitter)

    enable_image=True;image_key="webp";enable_text=True
    prompt_length=[3,2000];bucket_ratio=None;min_resolution=384
    def filter_dataset(item):
        # print('preprocess begin...')
        # print(item['txt'],item['json'],item['__key__'],item['__url__'])
        # pdb.set_trace()
        decision = 0
        if enable_image and image_key not in item:
            decision = 1
        if 'json' in item:
            infos = json.loads(item['json'])
            if enable_text and item['__key__'] not in prompt_infos:#没有llava recap的项
                prompt_cur=infos.get('caption',None)#有一些奇奇怪怪国家的语言，countchinese改成countenglish
                if infos['LANGUAGE']=='en' and len(prompt_cur)>prompt_length[0] and len(prompt_cur)<prompt_length[1]:
                    decision = 0
                else:
                    decision = 2
            if 'pwatermark' in infos and infos['pwatermark'] is not None and infos['pwatermark'] > 0.6:
                decision = 3 
            if 'punsafe' in infos and infos['punsafe'] is not None and infos['punsafe'] > 0.5:
                decision = 4
            
            if 'width' in infos and 'height' in infos and infos['height'] and infos['width'] and max(infos['height'], infos['width']) / min(infos['height'], infos['width']) > 3.0:
                decision = 5 
            
            if 'width' in infos and 'height' in infos and infos['height'] and infos['width'] and bucket_ratio is not None:
                rs = infos['height']/infos['width']
                if rs <= bucket_ratio[0] or rs > bucket_ratio[1]:
                    decision = 6 
                
            if 'width' in infos and infos['width'] and infos['width'] < min_resolution:
                decision = 7 
            if 'height' in infos and infos['height'] and infos['height'] < min_resolution:
                decision = 8
            # if infos['similarity']<0.25:
            #     print(infos['similarity'],infos['url'],prompt_cur)
        else:
            decision = 9 
        return True if decision==0 else False

    count=0
    filtered_dataset = dataset.select(filter_dataset)
    for _ in filtered_dataset:
        count+=1
        if count%10==0:
            print(count)
    print('total ',count)
    # pdb.set_trace()
    # dataset = create_webdataset(web_tar_list,
    #     size=512,
    #     enable_text=True,
    #     enable_image=True,
    #     image_key="jpg",
    #     caption_key="txt",
    #     cache_path=None,
    #     min_resolution=512)

    # train_dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=2,
    #         shuffle=False,
    #         collate_fn=collate_fn,
    #         num_workers=num_prepro_workers,
    #         pin_memory=True,
    #         prefetch_factor=2,
    #     )
    # pdb.set_trace()
    # # for data in train_dataloader:
    # #     print(data['pixel_values'].shape, data['input_ids'].shape)
    # # embed()
