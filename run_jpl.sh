#!/usr/bin/env bash

gpu_array=($LWLL_TA1_GPUS)
if [[ $LWLL_TA1_GPU == "all" ]]; then
    sed -i "/num_processes:/c\num_processes: $(nvidia-smi --list-gpus | wc -l)" accelerate_config.yml
else
    sed -i "/num_processes:/c\num_processes: ${#gpu_array[@]}" accelerate_config.yml
fi

accelerate launch --config_file ./accelerate_config.yml ./run_jpl.py --mode prod --simple_version false --folder evaluation