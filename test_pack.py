import os, sys 
from tools.datasets.pack import packer


def test_pack():
    """
    example：
    image name：000000581831.png
    json name：000000581831.json
    json info：
    {
        "id": "000000581831-2975",
        "basename": "000000581831.png",
        "prompt": "A group of small children in matching shirts with two soccer balls.",
        "height": 1224,
        "width": 1224
    }
    """
    data_dir = '/home/nfs/nfs-136/artstation/part1_cleaned'
    dataset_name = 'artstationPart1clean'
    image_dir_name = 'images' 
    json_dir_name = 'jsons'
    save_dir = '/home/nfs/nfs-133/artstationPart1clean_packs/'

    packer.common_packer(
        data_dir=data_dir,
        image_dir_name=image_dir_name,
        json_dir_name=json_dir_name,
        dataset_name=dataset_name,
        save_dir=save_dir,
        num_per_package=10000,
    )


if __name__ == '__main__':
    test_pack()
