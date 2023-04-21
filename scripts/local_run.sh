#!/bin/bash

for dataset_dir in '/Users/menga/Desktop/github/zsl_taglets/development'; do
for vis_encoder in 'ViT-B/32'; do 
for split_seed in 500; do 
for dataset_name in RESICS45; do
for model in multimodal_fpl; do 
for optim_seed in 1; do 
    
    export OPTIM_SEED="$optim_seed"
    export VIS_ENCODER="$vis_encoder"
    export DATASET_NAME="$dataset_name"
    export SPLIT_SEED="$split_seed"
    export MODEL="$model"
    export DATASET_DIR="$dataset_dir"

    accelerate launch --config_file ./accelerate_localtest_config.yml ./run_main_ssl.py \
                    --model_config ${model}_config.yml --learning_paradigm ssl

done
done
done
done
done
done