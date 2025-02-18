from diffusers import StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline,AutoencoderKL,EulerAncestralDiscreteScheduler
from PIL import Image
import os
import json
import torch

def prepare_sdxl_official():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path='/home/disk2/mxx/datasets/image_reward/benchmark-generations/sdxl'
    if not os.path.exists(save_path):os.makedirs(save_path)
    base = StableDiffusionXLPipeline.from_pretrained(
        "/home/disk1/liangtao/comm/stable-diffusion-xl-base-1.0"
    ).to(device)
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "/home/disk2/mxx/ckpts/stable-diffusion-xl-refiner-1.0"
    ).to(device)
    print('loaded pipe...')

    # load prompt samples
    sample_per_batch=10
    prompt_with_id_list = []
    with open('/home/disk2/mxx/VAR/metrics/image_reward/benchmark/benchmark-prompts.json', "r") as f:
        prompt_with_id_list = json.load(f)
    num_prompts = len(prompt_with_id_list)

    for item in prompt_with_id_list:
        prompt_id = [item["id"]]*sample_per_batch
        prompt = [item["prompt"]]*sample_per_batch
        print(item["prompt"])

        image = base(
        prompt=prompt,
        num_inference_steps=40,
        denoising_end=0.8,
        output_type="latent",
        height=512,width=512
        ).images
        image_B3HW = refiner(
            prompt=prompt,
            num_inference_steps=40,
            denoising_start=0.8,
            image=image,
            height=512,width=512
        ).images

        for i,label in enumerate(prompt_id):
            image_B3HW[i].save(os.path.join(save_path,f"{label}_{i}.png"))
            print(os.path.join(save_path,f"{label}_{i}.png"),' evaluated and saved...')


def prepare_sdxl_512():
    # 非官方512微调版本
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path='/home/disk2/mxx/datasets/image_reward/benchmark-generations/sdxl_512'
    if not os.path.exists(save_path):os.makedirs(save_path)
    base = StableDiffusionXLPipeline.from_pretrained(
        "/home/disk2/mxx/ckpts/SDXL-512",
        use_safetensors=True
    ).to(device)
    print('loaded pipe...')
    base.scheduler = EulerAncestralDiscreteScheduler.from_config(base.scheduler.config)

    # load prompt samples
    sample_per_batch=10
    prompt_with_id_list = []
    with open('/home/disk2/mxx/VAR/metrics/image_reward/benchmark/benchmark-prompts.json', "r") as f:
        prompt_with_id_list = json.load(f)
    num_prompts = len(prompt_with_id_list)

    for item in prompt_with_id_list:
        prompt_id = [item["id"]]*sample_per_batch
        prompt = [item["prompt"]]*sample_per_batch
        print(item["prompt"])

        image_B3HW = base(
            prompt=prompt,
            num_inference_steps=40,
            height=512,width=512,
            target_size=(1024, 1024),
            original_size=(4096, 4096)
        ).images

        for i,label in enumerate(prompt_id):
            image_B3HW[i].save(os.path.join(save_path,f"{label}_{i}.png"))
            print(os.path.join(save_path,f"{label}_{i}.png"),' evaluated and saved...')

if __name__=="__main__":
    prepare_sdxl_512()