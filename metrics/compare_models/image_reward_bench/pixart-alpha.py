from diffusers import PixArtAlphaPipeline
from PIL import Image
import os
import json
import torch
import pdb


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    repo_id = "/home/disk1/liangtao/comm/PixArt-alpha--PixArt-XL-2-512x512"
    save_path='/home/disk2/mxx/datasets/image_reward/benchmark-generations/pixart_alpha'
    if not os.path.exists(save_path):os.makedirs(save_path)
    pipe = PixArtAlphaPipeline.from_pretrained(repo_id, use_safetensors=True).to(device)
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

        image_B3HW=pipe(prompt,height=512,width=512).images

        for i,label in enumerate(prompt_id):
            image_B3HW[i].save(os.path.join(save_path,f"{label}_{i}.png"))
            print(os.path.join(save_path,f"{label}_{i}.png"),' evaluated and saved...')