#!/bin/bash

python Experiment.py --max_iter 20000 --size 40 &

python Experiment.py --max_iter 20000 --size 20 &

wait
