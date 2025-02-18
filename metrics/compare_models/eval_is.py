import os
import numpy as np
import torch
from torchvision import models, transforms
from torch.nn.functional import softmax
from PIL import Image
import argparse

def load_inception_model():
    model = models.inception_v3(pretrained=False)
    local_inception_path='/home/disk2/nfs/maxiaoxiao/ckpts/inceptionv3_torch/inception_v3_google-0cc3c7bd.pth'
    state_dict=torch.load(local_inception_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def calculate_inception_score(images, model, splits=10):
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_probs = []
    for img in images:
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            preds = model(img_tensor)
            probs = softmax(preds, dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.vstack(all_probs)

    scores = []
    for i in range(splits):
        part = all_probs[i * (len(all_probs) // splits): (i + 1) * (len(all_probs) // splits)]
        kl_div = np.mean(np.sum(part * (np.log(part) - np.log(np.mean(all_probs, axis=0))), axis=1))
        scores.append(np.exp(kl_div))

    return np.mean(scores), np.std(scores)

def load_images_from_folders(prediction_folder, reference_folder):
    images = []
    for filename in os.listdir(prediction_folder):
        pred_path = os.path.join(prediction_folder, filename)
        ref_path = os.path.join(reference_folder, filename)
        
        if os.path.isfile(pred_path) and os.path.isfile(ref_path):
            pred_image = Image.open(pred_path).convert('RGB')
            ref_image = Image.open(ref_path).convert('RGB')
            images.append(pred_image)  # 你可以选择只用预测图像或两者都用
            images.append(ref_image)  # 如果你希望一起计算

    return images

# 使用argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Inception Score for images.')
    parser.add_argument('--prediction_folder', type=str, required=True, help='Path to the prediction_total folder.')
    parser.add_argument('--reference_folder', type=str, required=True, help='Path to the reference_total folder.')

    args = parser.parse_args()

    model = load_inception_model()

    images = load_images_from_folders(args.prediction_folder, args.reference_folder)
    is_mean, is_std = calculate_inception_score(images, model)
    print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
