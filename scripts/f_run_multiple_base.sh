#!/bin/bash

port=1276

for vis_encoder in 'ViT-B/32' 'ViT-L/14'; do 

    sbatch --export=vis_encoder=$vis_encoder,PORT=$port --output=logs/flowers102_vpt_base_methods.out scripts/test_vpt_run.sh 
    port=$((port+1))

    sleep $[ ( $RANDOM % 30 )  + 40 ]s
done
done