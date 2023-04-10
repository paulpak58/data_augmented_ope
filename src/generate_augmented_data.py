# Inspired by https://github.com/Badger-RL/GuidedDataAugmentationForRobotics/blob/main/src/generate/generate_augmented_dataset.py
# Utils from https://github.com/Badger-RL/GuidedDataAugmentationForRobotics/blob/main/src/generate/utils.py
# Revised by Paul Pak

import os
import argparse
import h5py
import gym
import numpy as np

# File imports
from augment import RotateReflectTranslate
from utils import check_valid

def reset_data():
    return {'observations': [], 'next_observations': [], 'actions': [], 'rewards': [], 'terminals': [], 'truncated': [],}

def append_data(data, state, action, reward, next_state, done):
    data['observations'].append(state)
    data['next_observations'].append(next_state)
    data['actions'].append(action)
    data['rewards'].append(reward)
    data['terminals'].append(done)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--observed-dataset-path', type=str, default=None)
    parser.add_argument('--policy', type=str, default='expert', help='Type of policy used to generate the observed dataset')
    parser.add_argument('--augmentation-ratio', '-aug-ratio', type=int, default=1, help='Number of augmentations per observed transition')
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--save-name', type=str, default=None)
    args = parser.parse_args()

    aug_ratio = args.augmentation_ratio
    policy = args.policy

    env_name = 'halfcheetah-medium-v0'
    d4rl_env = gym.make(env_name)
    d4rl_dataset = d4rl_env.get_dataset()
    observed_dataset = d4rl_dataset
    print(f'observed_dataset.keys(): {observed_dataset.keys()}')
    print(f'Length of observed_dataset: {len(observed_dataset["observations"])}')



    # observed_data_hdf5 = h5py.File(f"{args.observed_dataset_path}", "r")
    # observed_dataset = {key: observed_data_hdf5[key][()] for key in observed_data_hdf5.keys()}
    n = observed_dataset['observations'].shape[0]
    env = gym.make('PushBallToGoal-v0')
    f = RotateReflectTranslate(env=None)

    # Create augmented buffer
    aug_dataset = reset_data()
    aug_count = 0
    invalid_count = 0
    i = 0
    while aug_count < n*aug_ratio:
        for _ in range(aug_ratio):
            idx = i%n
            aug_obs, aug_next_obs, aug_action, aug_reward, aug_done = f.augment(
                observed_dataset['observations'][idx],
                observed_dataset['next_observations'][idx],
                observed_dataset['actions'][idx],
                observed_dataset['rewards'][idx],
                observed_dataset['terminals'][idx]
            )
            i += 1
            if aug_obs is not None:
                is_valid = check_valid(env=env, aug_obs=[aug_obs], aug_action=[aug_action], aug_reward=[aug_reward], aug_next_obs=[aug_next_obs])
                if is_valid:
                    aug_count += 1
                    append_data(aug_dataset, aug_obs, aug_action, aug_reward, aug_next_obs, aug_done)
                else:
                    invalid_count += 1
            if aug_count >= n*aug_ratio:
                break
    print(f'Invalid Count: {invalid_count}')
    os.makedirs(args.save_dir, exist_ok=True)
    new_dataset = h5py.File(f"{args.save_dir}/{args.save_name}", "w")
    for key in aug_dataset:
        observed = observed_dataset[key]
        augmented = np.array(aug_dataset[key])
        data=np.concatenate([observed, augmented])
        data = data.astype(np.bool_) if key in ['terminals','timeouts'] else data.astype(np.float32)
        new_dataset.create_dataset(key, data=data, compression="gzip")
    print(f'Augmented count: {aug_count}')
    print(f'Augmented dataset size: {len(aug_dataset["observations"])}')

