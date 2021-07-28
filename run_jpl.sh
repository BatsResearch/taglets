#!/usr/bin/env bash

accelerate launch --config_file ./accelerate_config.yml ./run_jpl.py --batch_size 64 --mode dev --simple_version false --folder development
