#!/usr/bin/env bash

# Increases open file limit for multiprocessing
ulimit -n 65536
export PYTHONPATH=".:$PYTHONPATH"
python taglets/task/JPL.py
