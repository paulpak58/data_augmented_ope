import gym
import d4rl 

DRL_SUPPRESS_IMPORT_ERROR = 1

# Create the environment
env = gym.make('maze2d-umaze-v0')

# Automatically download and return the dataset
dataset = env.get_dataset()
print(dataset['observations']) # An (N, dim_observation)-dimensional numpy array of observations
print(dataset['actions']) # An (N, dim_action)-dimensional numpy array of actions
print(dataset['rewards']) # An (N,)-dimensional numpy array of rewards
