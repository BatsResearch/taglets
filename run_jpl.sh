#!/usr/bin/env bash

# Increases open file limit for multiprocessing
ulimit -n 1000000
export PYTHONPATH=".:$PYTHONPATH"
python -m taglets.task.jpl --mode dev
