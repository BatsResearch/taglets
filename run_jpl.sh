#!/usr/bin/env bash

# Increases open file limit for multiprocessing
ulimit -n 1000000
export PYTHONPATH=".:$PYTHONPATH"
python -m taglets.task.jpl --dataset_dir /lwll/development --scads_root_dir /lwll/external --task_ix 0
