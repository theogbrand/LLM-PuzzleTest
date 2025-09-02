#!/bin/bash
set -e

MODEL_NAME=$1

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
  python main.py evaluate_multi_choice data/${DATA}.json \
  --model_name ${MODEL_NAME} \
  --prompt_name cot_multi_extract
done