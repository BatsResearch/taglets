#!/bin/bash

#SBATCH --mem=32G
#SBATCH -p gpu-he --gres=gpu:3
#SBATCH -t 1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH -o ../logs/2021-pseudoshots

module load python/3.7.4
module load cuda/9.2.148


source MyEnv/bin/activate

./run_jpl.sh


