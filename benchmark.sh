#!/bin/bash

ENVIRONMENTS=("CartPole-v1" "MountainCar-v0" "Acrobot-v1" "LunarLander-v3")
SEEDS=(0 1 2 3 4)

for ENV in "${ENVIRONMENTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running: env=${ENV}, seed=${SEED}"
        python ppo_jax.py \
            --num_iterations=500_000 \
            --num_steps=128 \
            --num_envs=8 \
            --update_epochs=10 \
            --seed=${SEED} \
            --env_name=${ENV} \
            --capture_video \
            --track
    done
done