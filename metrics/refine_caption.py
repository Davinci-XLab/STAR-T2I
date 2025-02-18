import os
import re
import json

def format_sentence(sentence):
    # Remove non-word characters except for spaces and hyphens
    clean_sentence = re.sub(r'[^\w\s-]', '', sentence)
    # Replace spaces with underscores
    underscore_sentence = re.sub(r'\s+', '_', clean_sentence)
    words = underscore_sentence.split('_')
    # Join the first 20 words with underscores
    limited_sentence = '_'.join(words[:20])
    return limited_sentence

def main(image_folder, caption_file, output_json,save_path,max_len=77):
    captions_dict = {}

    with open(caption_file, 'r') as f:
        captions = f.readlines()

    for i, caption in enumerate(captions):
        # Format the sentence
        formatted_caption = format_sentence(caption.strip())

        # Load image
        image_name = f"{formatted_caption}.png"
        image_path = os.path.join(image_folder, image_name)

        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist. Skipping...")
            continue

        if not os.path.exists(save_path):os.makedirs(save_path)
        with open(os.path.join(save_path,'%s.txt'%formatted_caption),"w") as f:
            if len(caption.split(' '))>max_len:
                caption=' '.join(caption[:max_len])
            f.write(caption)

    # # Write captions dictionary to JSON file
    # with open(output_json, 'w') as json_file:
    #     json.dump(captions_dict, json_file, indent=4)

if __name__ == "__main__":
    image_folder = "../ep16_iter0"
    caption_file = "../dataset/prompts/benchmark_v3_en.txt"
    output_json = "caption_benchmark_v3_en.json"
    save_path='./captions_ep16'
    main(image_folder, caption_file, output_json,save_path)
