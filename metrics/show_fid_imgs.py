import os
import random
from PIL import Image
import argparse


def combine_images(reference_folder, prediction_folder, output_folder):
    # 获取reference和prediction文件夹中的所有文件名
    reference_files = os.listdir(reference_folder)
    prediction_files = os.listdir(prediction_folder)

    # 确保两个文件夹中的文件名是一一对应的
    reference_files.sort()
    prediction_files.sort()

    # 从reference文件夹中随机选择20张图片
    selected_reference_files = random.sample(reference_files, 20)

    for ref_file in selected_reference_files:
        # 找到对应的prediction文件
        prediction_file = ref_file
        if prediction_file in prediction_files:
            # 打开reference和prediction图片
            ref_image = Image.open(os.path.join(reference_folder, ref_file))
            pred_image = Image.open(os.path.join(prediction_folder, prediction_file))

            # 确保两张图片大小相同
            if ref_image.size == pred_image.size:
                # 创建一个新的图像，左边是reference，右边是prediction
                combined_image = Image.new('RGB', (ref_image.width * 2, ref_image.height))
                combined_image.paste(ref_image, (0, 0))
                combined_image.paste(pred_image, (ref_image.width, 0))

                # 保存合并后的图像到指定路径下
                if not os.path.exists(output_folder):os.makedirs(output_folder)
                combined_image.save(os.path.join(output_folder, ref_file))
            else:
                print(f"Image sizes do not match for {ref_file}. Skipping...")
        else:
            print(f"Corresponding prediction image not found for {ref_file}. Skipping...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine reference and prediction images side by side.")
    parser.add_argument("--reference_folder", type=str, help="Path to reference images folder")
    parser.add_argument("--prediction_folder", type=str, help="Path to prediction images folder")
    parser.add_argument("--output_folder", type=str, help="Path to output folder")

    args = parser.parse_args()
    combine_images(args.reference_folder, args.prediction_folder, args.output_folder)
