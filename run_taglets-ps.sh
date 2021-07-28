#!/bin/bash

#SBATCH --mem=20G
#SBATCH -p gpu --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o ../logs/2021-pseudoshots-fixmatch-zslkg

module load python/3.7.4
module load cuda/9.2.148


source env/bin/activate

./run_jpl.sh


