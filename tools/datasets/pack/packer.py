# coding=utf-8
import os
import json
import multiprocessing
import shutil
import time
import nanoid
import string
from glob import glob
from PIL import Image
from io import BytesIO
from tools.datasets.lmdb_dict import LMDBDict


def make_lmdb(save_dir, img_list, jobs=None, dataset_name=None, delete_original=True):
    db_name = dataset_name+"-{}".format(nanoid.generate(string.digits + string.ascii_lowercase, size=6))
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{now}]{db_name} start task ...")

    db_path = os.path.join(save_dir, db_name)
    info_save = os.path.join(save_dir, db_name + "-info.json")

    info_object = {}
    d = LMDBDict(db_path, mode="w")
    exception_count = 0
    for image_key, image_path, info in img_list:
        try:
            img = Image.open(image_path)
            with BytesIO() as f:
                img.save(f, format='png')
                img_byte = f.getvalue()
        except Exception as e:
            exception_count += 1
            continue
        info_object[image_key] = info
        d[image_key] = img_byte
    d.flush()
    del d
    json.dump(info_object, open(info_save, "w"), indent=4, ensure_ascii=False)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{now}]{db_name} end task ...")


def get_files_and_infos(data_dir, image_dir_name, dataset_name, json_paths):
    res = []
    exception_count = 0
    for json_path in json_paths:
        assert os.path.exists(json_path)
        try:
            with open(json_path, encoding='utf-8') as jfile:
                info = json.load(jfile)
            # basename = info['basename']
            basename = os.path.basename(json_path).replace('json', 'webp')
            prompt = info.get('google_jp')
            if prompt is None:
                prompt = info.get('llava_zh', None)
            if prompt is None:
                prompt = info.get('minicpm_caption', None)
            if prompt is None:
                prompt = info.get('google_jp', None)
            if prompt is None:
                prompt = info.get('caption', None)
            if prompt is None:
                prompt = info.get('swintag', None)
            if prompt is None:
                prompt = info.get('florence_en', None)
            if prompt is None:
                print('The image {} prompt is None'.format(basename))
                continue
            image_key = '{}-{}'.format(dataset_name, info['id'])
            image_path = os.path.join(data_dir, image_dir_name, basename)

            res.append((image_key, image_path, info.copy()))
        except KeyboardInterrupt:
            break
        except Exception as e:
            exception_count += 1
    print("Error task:", exception_count)
    return res, json_paths


def worker(task_queue, data_dir, image_dir_name, dataset_name, save_dir):
    while True:
        jobs, _ = task_queue.get()  # 每个进程不断的从队列中获取任务
        if jobs is None:
            print("Worker 退出")
            break
        res, res_jobs = get_files_and_infos(data_dir, image_dir_name, dataset_name, json_paths=jobs)
        if len(res) == 0:
            continue
        make_lmdb(save_dir, res, res_jobs, dataset_name, delete_original=True)


def common_packer(data_dir, image_dir_name, json_dir_name, dataset_name, save_dir, num_per_package=1000):
    os.makedirs(save_dir, exist_ok=True)   # 创建保存目录
    json_dir = os.path.join(data_dir, json_dir_name)   # 存放 json 文件的目录
    json_paths = glob(os.path.join(json_dir, '*.json'))   # 获取所有的 json 文件
    num_files = len(json_paths)   # json 文件个数
    num_package = num_files // num_per_package   # 计算需要打包的个数  # 0
    if num_files % num_per_package > 0:
        num_package += 1
    # num_work = min(num_package, multiprocessing.cpu_count())  # 计算需要的进程个数
    num_work = 4
    task_queue = multiprocessing.Queue()    # 创建任务队列
    # Add the tasks to the queue
    for i in range(num_package):
        start = i * num_per_package
        if i == num_package - 1:
            end = num_files
        else:
            end = (i + 1) * num_per_package
        jobs = json_paths[start:end]  # 每个包中存放的任务
        task_queue.put((jobs, None))

    # Start process
    processes = []
    for i in range(num_work):
        process = multiprocessing.Process(target=worker,
                                          args=(task_queue, data_dir, image_dir_name, dataset_name, save_dir))
        process.start()
        processes.append(process)

    # Stop the worker processes
    for i in range(num_work * 2):
        task_queue.put((None, None))

    for process in processes:
        process.join()
