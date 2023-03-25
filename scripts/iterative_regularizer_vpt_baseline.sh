#!/bin/bash
#SBATCH --job-name=R-iter_meth
#SBATCH --output=logs/resics_iter_methods_split_500.out
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
for model in iterative_regularizer_vpt_baseline; do #  all_vpt_pseudo_baseline; do 
for optim_seed in 1 2 3 4 5; do # 10 100 50 400 250; do
for alpha in 1; do #0  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
for regularizer in kl ce both; do
for class_regularizer in seen all; do

    export OPTIM_SEED="$optim_seed"
    export VIS_ENCODER="$vis_encoder"
    export DATASET_NAME="$dataset_name"
    export SPLIT_SEED="$split_seed"
    export MODEL="$model"
    export ALPHA="$alpha"
    export REGULARIZER="$regularizer"
    export CLASSES_REGULARIZER="$class_regularizer"
    
    sed -i 's/^\(\s*main_process_port\s*:\s*\).*/\12075/'  accelerate_config.yml
    accelerate launch --config_file ./accelerate_config.yml ./run_main.py \
                    --model_config ${model}_config.yml
done
done
done
done
done
done
done
done