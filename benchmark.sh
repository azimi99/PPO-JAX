#!/bin/bash

ENVIRONMENTS=("CartPole-v1" "MountainCar-v0" "Acrobot-v1" "LunarLander-v3")
SEEDS=(0 1 2 3 4)

for ENV in "${ENVIRONMENTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running: env=${ENV}, seed=${SEED}"
        if [ "$ENV" = "LunarLander-v3" ]; then
            python ppo_jax.py \
            --total_timesteps=1500000 \
            --num_steps=512 \
            --num_envs=8 \
            --update_epochs=10 \
            --learning_rate=6e-3 \
            --seed=${SEED} \
            --env_name=${ENV} \
            --capture_video \
            --track 
        # else
        #     python ppo_jax.py \
        #     --tota_timesteps=500_000 \
        #     --num_steps=128 \
        #     --num_envs=8 \
        #     --update_epochs=10 \
        #     --learning_rate=3e-4 \
        #     --seed=${SEED} \
        #     --env_name=${ENV} \
        #     --capture_video \
        #     --track 
        fi
    done
done