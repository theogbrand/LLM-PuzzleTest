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
  "circle_size_number"
  "color_grid"
  "color_hexagon"
  "color_number_hexagon"
  "color_overlap_squares"
  "color_size_circle"
  "grid_number_color"
  "grid_number"
  "polygon_sides_color"
  "polygon_sides_number"
  "rectangle_height_color"
  "rectangle_height_number"
  "shape_morph"
  "shape_reflect"
  "shape_size_grid"
  "shape_size_hexagon"
  "size_cycle"
  "size_grid"
  "triangle"
  "venn"
)

# Loop through each dataset and run the evaluation
for DATA in "${DATASETS[@]}"; do
  echo "Evaluating dataset: $DATA with model: $MODEL_NAME"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py evaluate_multi_choice data/${DATA}.json \
  --model_name ${MODEL_NAME} \
  --prompt_name cot_multi_extract
done