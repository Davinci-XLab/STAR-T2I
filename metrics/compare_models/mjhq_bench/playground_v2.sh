#!/bin/bash
set -x

#!/bin/bash

# List of categories
categories=(
    "animals"
    "art"
    "fashion"
    "food"
    "indoor"
    "landscape"
    "logo"
    "people"
)

# Total number of GPUs
num_gpus=8

/opt/miniconda3/bin/python -m pip install diffusers==0.31.0 accelerate transformers==4.44.2

# Directory paths (replace these with your actual paths)
savedir_pred="/path/to/pred"
savedir_gt="/path/to/gt"
# cd /nfs-26/maxiaoxiao/VAR/metrics/compare_models/mjhq_bench
cd /nfs-26/maxiaoxiao/VAR_new/metrics/compare_models/mjhq_bench
chmod -R 777 /nfs-141/maxiaoxiao/eval_results

# Loop over categories
for i in "${!categories[@]}"; do
    category="${categories[$i]}"
    gpu_id=$((i % num_gpus))

    echo "Starting process for category $category on GPU $gpu_id"

    # Run the python script in the background with the specified GPU
    # CUDA_VISIBLE_DEVICES=$gpu_id /opt/miniconda3/bin/python playground_v2.py --category "$category" &
        CUDA_VISIBLE_DEVICES=$gpu_id /opt/miniconda3/bin/python flux.py --category "$category" &
done

# 等待所有后台进程完成
wait
echo "All processes completed."
# CUDA_VISIBLE_DEVICES=7 /opt/miniconda3/bin/python sdxl_.py --category "vehicles"
# CUDA_VISIBLE_DEVICES=7 /opt/miniconda3/bin/python pixart_sigma.py --category "vehicles"
# CUDA_VISIBLE_DEVICES=7 /opt/miniconda3/bin/python flux.py --category "vehicles"