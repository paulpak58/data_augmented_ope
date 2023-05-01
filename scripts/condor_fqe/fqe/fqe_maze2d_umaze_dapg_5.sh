#!/bin/bash

# activate necessary packages from docker container
source /home/paulpak/.bashrc
export D4RL_DATASET_DIR=./datasets

# copy datasets from staging dir
cp /staging/pepak/datasets.tar.gz ./
tar -xzvf datasets.tar.gz

# extract source files
tar -xzvf data_augmented_ope.tar.gz
home=./data_augmented_ope
env_name=maze2d-umaze-v1
dir_name=maze2d_umaze
behavioral_policy=maze2d_umaze_dapg_5.pkl
algo=fqe
results_dir=./results/{dir_name}/

# create results dir
if [ ! -d ${results_dir} ]; then
    mkdir -p ${results_dir}
fi

# see num_mc_episodes & num_updates
python3 ${home}/data_augmented_train_eval.py \
    --env_name=${env_name} \
    --d4rl_policy_filename=${home}/d4rl/${dir_name}/${behavioral_policy} \
    --num_mc_episodes=256 --num_updates=250000 \
    --algo=${algo} --d4rl --target_policy_std=0.0 \
    --logtostderr --noise_scale=0.0 --nobootstrap \
    --save_dir=${results_dir}

# copy results to top-level dir
tar -czvf results.tar.gz ./results
cp results.tar.gz /staging/pepak/results.tar.gz

# remove copied source files
rm -rf data_augmented_ope
rm data_augmented_ope.tar.gz

# remove files from working dir
rm datasets.tar.gz
rm -rf datasets