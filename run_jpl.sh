#!/usr/bin/env bash

accelerate launch --config_file ./accelerate_config.yml ./run_jpl.py --mode prod --simple_version false --folder evaluation