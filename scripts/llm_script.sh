#!/bin/bash

# Enable CUDA launch blocking for debugging
export CUDA_LAUNCH_BLOCKING=1

# Define directories
FOLDERS=("k_all_c_all")
CASES=("cold"  "warm")

# Loop through each directory and case
for FOLDER in "${FOLDERS[@]}"; do
    for CASE in "${CASES[@]}"; do
        echo "Running inference for directory: $FOLDER, case: $CASE"
        python -m inference.inference_zeroshot  --case "$CASE"  --folder "$FOLDER"
        python -m inference.inference_icl --case "$CASE"  --folder "$FOLDER"
        python -m inference.inference_instruct --case "$CASE"  --folder "$FOLDER"
    done
done
