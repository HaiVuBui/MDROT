#!/bin/bash

python Experiment_mnist.py --max_iter 100000 &

python Experiment.py --max_iter 100000 --size 60 &

#python Experiment.py --max_iter 100000 --size 40 &

#python Experiment.py --max_iter 100000 --size 20 &

wait
