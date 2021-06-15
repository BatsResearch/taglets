#!/usr/bin/env bash

accelerate launch --config_file ./accelerate_config.yml ./run_jpl.py --mode dev --simple_version false --folder development