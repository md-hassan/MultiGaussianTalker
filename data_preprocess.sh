#!/bin/bash

YOUR_DATASET_DIR="/workspace/code/MultiGaussianTalker/data/full_vids"

DATASET_NAMES=("may" "purple_girl" "black_man")  # Add all your dataset names here
# DATASET_NAMES=("pewdiepie")

cd data_utils/face_tracking

python convert_BFM.py
wait

cd ../../

for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    python data_utils/process.py "${YOUR_DATASET_DIR}/${DATASET_NAME}/${DATASET_NAME}.mp4"
    wait
done