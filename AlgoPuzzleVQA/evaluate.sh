#!/bin/bash
set -e

MODEL_NAME=$1

# Find number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found $NUM_GPUS GPUs available"

# Set CUDA_VISIBLE_DEVICES to use all available GPUs
if [ $NUM_GPUS -gt 0 ]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
else
    echo "No GPUs found, running on CPU"
    CUDA_VISIBLE_DEVICES=""
fi

# List of datasets
DATASETS=(
  "board_tile"
  "calendar"
  "chain_link"
  "checker_move"
  "clock"
  "colour_hue"
  "map"
  "maze"
  "move_box"
  "n_queens"
  "number_slide"
  "rotting_kiwi"
  "rubiks_cube"
  "think_dot"
  "tower_of_hanoi"
  "water_jugs"
  "wheel_of_fortune"
  "wood_slide"
)

# Loop through each dataset and run the evaluation
for DATA in "${DATASETS[@]}"; do
  echo "Evaluating dataset: $DATA with model: $MODEL_NAME"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py evaluate_multi_choice data/${DATA}.json \
  --model_name ${MODEL_NAME} \
  --prompt_name cot_multi_extract
done