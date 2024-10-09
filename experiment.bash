#!/bin/bash

# Hyperparameter ranges
learning_rates=(0.00001 0.00005 0.0001 0.0003 0.001)

# Iterate over each combination of hyperparameters
for i in {1..5}; do
    # Run the Python script with the current hyperparameter combination
    python baselines/main.py --learning_rate 0.001 --num_ood_actions 30
done
