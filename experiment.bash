#!/bin/bash

# Number of times to run the command
num_runs=5

# Loop to run the command multiple times
for i in $(seq 5 $num_runs); do
    echo $i
    command="python baselines/main.py --seed=$i --system=qmix+cql --dataset=Good --scenario=2s3z"
    $command
done
