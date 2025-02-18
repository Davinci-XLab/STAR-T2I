from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import json
import pdb

if __name__ == "__main__":
    repo_id = "/home/disk1/liangtao/comm/stable-diffusion-v2-1"
    save_path='/home/disk2/mxx/datasets/image_reward/benchmark-generations/sd2-1'
    if not os.path.exists(save_path):os.makedirs(save_path)
    pipe = StableDiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
    print('loaded pipe...')

    # load prompt samples
    sample_per_batch=8
    prompt_with_id_list = []
    with open('../image_reward/benchmark.json', "r") as f:
        prompt_with_id_list = json.load(f)
    num_prompts = len(prompt_with_id_list)

    for item in prompt_with_id_list:
        prompt_id = item["id"]
        prompt = item["prompt"]*10

        image_B3HW=pipe(prompt,height=512,width=512).images[0]
        pdb.set_trace()

        for i in range(sample_per_batch):
            image_B3HW.save(os.path.join(save_path,f"{prompt_id}_{i}.png"))


        