#!/usr/bin/env bash

accelerate launch --config_file ./accelerate_localtest_config.yml ./run_jpl.py --mode dev --folder development --model_config vpt_pseudo_disambiguate_config.yml