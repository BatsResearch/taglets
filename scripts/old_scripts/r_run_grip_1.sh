#!/bin/bash
#SBATCH --job-name=R-grip-1
#SBATCH --output=logs/recsis45_grip_1.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 96:00:00
#SBATCH -p gpu --gres=gpu:4
#SBATCH --exclude=gpu[717-718,1201-1204,1209,1403]
#SBATCH -C quadrortx

module load cuda/11.1.1

pwd
source ../../zsl/bin/activate

sleep $[ ( $RANDOM % 30 )  + 1 ]s


for vis_encoder in 'ViT-B/32'; do # 'ViT-B/32' 'RN50' 'ViT-L/14' 'RN101'
for split_seed in 500; do #  0 200
for dataset_name in RESICS45; do
for model in ablation_teacher_student; do
for optim_seed in 1 2 3 4 5; do

    export OPTIM_SEED="$optim_seed"
    export VIS_ENCODER="$vis_encoder"
    export DATASET_NAME="$dataset_name"
    export SPLIT_SEED="$split_seed"
    export MODEL="$model"
    
    sed -i 's/^\(\s*main_process_port\s*:\s*\).*/\12073/'  accelerate_config.yml
    accelerate launch --config_file ./accelerate_config.yml ./run_main.py \
                    --model_config ${model}_config.yml
done
done
done
done
done