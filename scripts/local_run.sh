#!/bin/bash


for vis_encoder in 'ViT-B/32'; do 
for split_seed in 500; do 
for dataset_name in RESICS45; do
for model in grip_visual; do 
for optim_seed in 1; do 
    
    export OPTIM_SEED="$optim_seed"
    export VIS_ENCODER="$vis_encoder"
    export DATASET_NAME="$dataset_name"
    export SPLIT_SEED="$split_seed"
    export MODEL="$model"

    accelerate launch --config_file ./accelerate_localtest_config.yml ./run_main_ul.py \
                    --model_config ${model}_config.yml --learning_paradigm ul

done
done
done
done
done