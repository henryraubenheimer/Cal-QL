#!/bin/bash

# Number of times to run the command
num_runs=10

# Loop to run the command multiple times
for i in $(seq 6 $num_runs); do
    command="python baselines/main.py --system=idrqn+cql --dataset=Poor --seed=$i"
    $command
done
