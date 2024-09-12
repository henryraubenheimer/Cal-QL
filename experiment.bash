#!/bin/bash

# Command to run
command="python baselines/main.py --system=idrqn+cql --dataset=Medium"

# Number of times to run the command
num_runs=4

# Loop to run the command multiple times
for i in $(seq 1 $num_runs); do
    echo "Run #$i"
    $command
done
