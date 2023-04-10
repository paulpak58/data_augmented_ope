#!/bin/bash

home=/Users/paul/Downloads/data_augmented_ope
D4RL_SUPPRESS_IMPORT_ERROR=1 python ${home}/augment/data_augmented_train_eval.py \
    --logtostderr \
    --d4rl \
    --env_name=antmaze-medium-play-v2 \
    --d4rl_policy_filename=./d4rl/antmaze_medium/antmaze_medium_dapg_0.pkl \
    --target_policy_std=0.0 \
    --num_mc_episodes=25 \
    --nobootstrap \
    --algo=fqe \
    --noise_scale=0.0 \
    --num_updates=10000