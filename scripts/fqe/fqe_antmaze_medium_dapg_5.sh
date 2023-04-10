#!/bin/bash

home=/Users/paul/Downloads/data_augmented_ope
D4RL_SUPPRESS_IMPORT_ERROR=1 python ${home}/src/data_augmented_train_eval.py \
    --logtostderr \
    --d4rl \
    --env_name=antmaze-medium-play-v1 \
    --d4rl_policy_filename=${home}/src/d4rl/antmaze_medium/antmaze_medium_dapg_5.pkl \
    --target_policy_std=0.0 \
    --num_mc_episodes=25 \
    --nobootstrap \
    --algo=fqe \
    --noise_scale=0.0 \
    --num_updates=10000 \
    --save_dir=${home}/results/antmaze_medium_dapg_5/