#!/usr/bin/env bash

accelerate launch --config_file ./accelerate_localtest_config.yml ./run_jpl.py --mode dev --folder development --model_config cocoop_baseline_config.yml
