#!/bin/bash

port=1270

for type in after_norm_prefix before_norm_cls before_norm_prefix; do # after_norm_cls
for prefix_size in 16 32; do

    sbatch --export=TYPE=$type,PREFIX_SIZE=$prefix_size,PORT=$port --output=logs/test_vpt_${type}_${prefix_size}.out scripts/test_vpt_run.sh 
    port=$((port+1))

    sleep $[ ( $RANDOM % 30 )  + 40 ]s
done
done