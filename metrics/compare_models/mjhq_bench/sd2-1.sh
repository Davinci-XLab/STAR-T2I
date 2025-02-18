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
    "plants"
)

# Total number of GPUs
num_gpus=8

# Directory paths (replace these with your actual paths)
savedir_pred="/path/to/pred"
savedir_gt="/path/to/gt"

# Loop over categories
for i in "${!categories[@]}"; do
    category="${categories[$i]}"
    gpu_id=$((i % num_gpus))

    echo "Starting process for category $category on GPU $gpu_id"

    # Run the python script in the background with the specified GPU
    CUDA_VISIBLE_DEVICES=$gpu_id python sd2-1.py --category "$category" &
done

CUDA_VISIBLE_DEVICES=$gpu_id python sd2-1.py --category "vehicles"

wait