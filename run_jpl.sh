#!/usr/bin/env bash

gpu_array=($LWLL_TA1_GPUS)
sed -i "/num_processes:/c\num_processes: ${#gpu_array[@]}" accelerate_config.yml

accelerate launch --config_file ./accelerate_config.yml ./run_jpl.py --mode prod --simple_version false --folder evaluation