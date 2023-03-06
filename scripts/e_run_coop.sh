#!/bin/bash
#SBATCH --job-name=E-coop
#SBATCH --output=logs/eurosat_coop.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -p gpu --gres=gpu:4
##SBATCH --exclude=gpu[717-718,1201-1204,1209,1403]

module load cuda/11.1.1

pwd
source ../../zsl/bin/activate

sleep $[ ( $RANDOM % 30 )  + 1 ]s

sed -i 's/^\(\s*main_process_port\s*:\s*\).*/\12072/'  accelerate_config.yml
for vis_encoder in 'RN50' 'RN101' 'ViT-B/32' 'ViT-L/14'; do 
for dataset_name in EuroSAT; do
for split_seed in 500 0 200; do
for optim_seed in 10 100 50 400 250; do

    export OPTIM_SEED="$optim_seed"
    export VIS_ENCODER="$vis_encoder"
    export DATASET_NAME="$dataset_name"
    export SPLIT_SEED="$split_seed"
    
    accelerate launch --config_file ./accelerate_config.yml ./run_main.py \
                    --model_config coop_baseline_config.yml
done
done
done
done