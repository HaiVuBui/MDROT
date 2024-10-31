#!/bin/bash

python Experiment_mnist.py --max_iter 50000 &

python Experiment.py --max_iter 50000 --size 60 &

wait
