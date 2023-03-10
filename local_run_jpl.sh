#!/usr/bin/env bash

for vis_encoder in 'ViT-B/32'; do # 'RN50' 'RN101' ' 'ViT-L/14''
for split_seed in 500; do
for optim_seed in 10; do
for model in coop_teacher_student; do 
for dataset_name in RESICS45; do

    export OPTIM_SEED="$optim_seed"
    export VIS_ENCODER="$vis_encoder"
    export DATASET_NAME="$dataset_name"
    export SPLIT_SEED="$split_seed"
    export MODEL="$model"
    
    accelerate launch --config_file ./accelerate_localtest_config.yml ./run_main.py \
                      --model_config local_${model}_config.yml

done
done
done
done
done
