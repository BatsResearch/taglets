#!/bin/bash


for vis_encoder in 'ViT-B/32'; do # 'ViT-B/32' 'RN50' 'ViT-L/14' 'RN101'
for split_seed in 500; do #  0 200
for dataset_name in RESICS45; do
for model in after_combo_vpt_baseline; do #  all_vpt_pseudo_baseline; do 
for optim_seed in 1; do # 2 3 4 5; do # 10 100 50 400 250; do
for alpha in 0.5; do #0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do

    export OPTIM_SEED="$optim_seed"
    export VIS_ENCODER="$vis_encoder"
    export DATASET_NAME="$dataset_name"
    export SPLIT_SEED="$split_seed"
    export MODEL="$model"
    export ALPHA="$alpha"
    
    #sed -i 's/^\(\s*main_process_port\s*:\s*\).*/\12078/'  accelerate_config.yml
    accelerate launch --config_file ./accelerate_localtest_config.yml ./run_main.py \
                    --model_config ${model}_config.yml
done
done
done
done
done
done