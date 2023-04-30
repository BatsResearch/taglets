#!/bin/bash
#SBATCH --job-name=RTR-FPL
#SBATCH --output=logs/trzsl_resics_fpl_2.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH -t 48:00:00
#SBATCH -p gpu --gres=gpu:4
##SBATCH --exclude=gpu[717-718,1201-1204,1209,1403]

module load cuda/11.1.1

pwd
source ../../zsl/bin/activate

sleep $[ ( $RANDOM % 30 )  + 40 ]s

for dataset_dir in '/users/cmenghin/data/bats/datasets/classification' ; do
for vis_encoder in 'ViT-L/14'; do # 'ViT-B/32'  'RN50' 'ViT-L/14' 'RN101'
for split_seed in 500; do #  0 200
for dataset_name in RESICS45; do
for model in textual_fpl visual_fpl multimodal_fpl; do # 
for optim_seed in 1 2 3 4 5; do # 2 3 4 5; do #10 100 50 400 250; do
    
    export OPTIM_SEED="$optim_seed"
    export VIS_ENCODER="$vis_encoder"
    export DATASET_NAME="$dataset_name"
    export SPLIT_SEED="$split_seed"
    export MODEL="$model"
    export DATASET_DIR="$dataset_dir"

    sed -i 's/^\(\s*main_process_port\s*:\s*\).*/\12070/'  accelerate_config.yml
    accelerate launch --config_file ./accelerate_config.yml ./run_main_trzsl.py \
                    --model_config ${model}_config.yml --learning_paradigm trzsl
done
done
done
done
done
done