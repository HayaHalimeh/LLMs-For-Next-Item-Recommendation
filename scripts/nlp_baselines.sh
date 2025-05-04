#!/bin/bash

# Enable CUDA launch blocking for debugging
export CUDA_LAUNCH_BLOCKING=1

# Define directories
FOLDERS=( "k_all_c_all") 
CASES=("cold" "warm")

# Loop through each directory and case
for FOLDER in "${FOLDERS[@]}"; do
    for CASE in "${CASES[@]}"; do
        echo "Running inference for directory: $FOLDER, case: $CASE"
        python -m inference.inference_random --folder "$FOLDER" --case "$CASE"
        python -m inference.inference_embsim --folder "$FOLDER" --case "$CASE"
        python -m inference.inference_nsp --folder "$FOLDER" --case "$CASE"
        python -m inference.inference_item_cls --folder "$FOLDER" --case "$CASE"
    done
done

