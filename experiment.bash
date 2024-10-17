#!/bin/bash

for i in {1..5}; do
    python baselines/main.py --offline_training_steps=50000 --seed=$i --system=idrqn+cql --online_cql_weight=2 --suffix=online_regularisation
done