#!/bin/bash
#SBATCH --job-name=R-base_meth
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


for vis_encoder in 'ViT-B/32'; do # 'ViT-B/32'  'RN50' 'ViT-L/14' 'RN101'
for split_seed in 500; do #  0 200
for dataset_name in RESICS45; do
for model in textual_prompt; do # coop_baseline
for optim_seed in 1; do # 2 3 4 5; do #10 100 50 400 250; do
    
    export OPTIM_SEED="$optim_seed"
    export VIS_ENCODER="$vis_encoder"
    export DATASET_NAME="$dataset_name"
    export SPLIT_SEED="$split_seed"
    export MODEL="$model"
    # export TYPE="$TYPE"
    # export PREFIX_SIZE="$PREFIX_SIZE"
    # #$PORT
    # echo $optim_seed
    # echo $TYPE
    # echo $PORT
    # $PORT
    sed -i 's/^\(\s*main_process_port\s*:\s*\).*/\12071/'  accelerate_config.yml
    accelerate launch --config_file ./accelerate_config.yml ./run_main.py \
                    --model_config ${model}_config.yml
done
done
done
done
done