#!/usr/bin/env bash

# Increases open file limit for multiprocessing
ulimit -n 65536
python taglets/task/JPL.py
