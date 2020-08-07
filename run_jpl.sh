#!/usr/bin/env bash

# Increases open file limit for multiprocessing
ulimit -n 65536
export PYTHONPATH=".:$PYTHONPATH"
python -m taglets.task.jpl --dataset_dir /lwll/development --task_ix 3
