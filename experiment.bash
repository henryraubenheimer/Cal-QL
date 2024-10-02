#!/bin/bash

# Number of times to run the command
num_runs=5

# Loop to run the command multiple times
for i in $(seq 1 $num_runs); do
    echo $i
    command="python baselines/main.py --seed=$i --scenario=8m"
    $command
done
