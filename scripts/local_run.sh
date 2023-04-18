#!/bin/bash


for vis_encoder in 'ViT-B/32'; do # 'ViT-B/32'  'RN50' 'ViT-L/14' 'RN101'
for split_seed in 500; do #  0 200
for dataset_name in RESICS45; do
for model in visual_prompt; do # coop_baseline
for optim_seed in 1; do # 2 3 4 5; do #10 100 50 400 250; do
    
    export OPTIM_SEED="$optim_seed"
    export VIS_ENCODER="$vis_encoder"
    export DATASET_NAME="$dataset_name"
    export SPLIT_SEED="$split_seed"
    export MODEL="$model"

    accelerate launch --config_file ./accelerate_localtest_config.yml ./run_main.py \
                    --model_config ${model}_config.yml

done
done
done
done
done