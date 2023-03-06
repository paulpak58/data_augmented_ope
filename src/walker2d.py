# Inspired from Nicholas Corrado's work
# Revised by Paul Pak
# https://github.com/NicholasCorrado/RL-Augment/blob/gym0.26.0/augment/rl/augmentation_functions/walker2d.py

import time
from typing import Dict, List, Any

import numpy as np

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction
from augment.simulate import simulate

'''
    """
    ### Description
    This environment builds on the hopper environment based on the work done by Erez, Tassa, and Todorov
    in ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf)
    by adding another set of legs making it possible for the robot to walker forward instead of
    hop. Like other Mujoco environments, this environment aims to increase the number of independent state
    and control variables as compared to the classic control environments. The walker is a
    two-dimensional two-legged figure that consist of four main body parts - a single torso at the top
    (with the two legs splitting after the torso), two thighs in the middle below the torso, two legs
    in the bottom below the thighs, and two feet attached to the legs on which the entire body rests.
    The goal is to make coordinate both sets of feet, legs, and thighs to move in the forward (right)
    direction by applying torques on the six hinges connecting the six body parts.
    ### Action Space
    The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.
    | Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    |-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
    | 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                        | hinge | torque (N m) |
    | 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                       | hinge | torque (N m) |
    | 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
    | 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
    | 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |
    ### Observation Space
    Observations consist of positional values of different body parts of the walker,
    followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.
    By default, observations do not include the x-coordinate of the top. It may
    be included by passing `exclude_current_positions_from_observation=False` during construction.
    In that case, the observation space will have 18 dimensions where the first dimension
    represent the x-coordinates of the top of the walker.
    Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x-coordinate
    of the top will be returned in `info` with key `"x_position"`.
    By default, observation is a `ndarray` with shape `(17,)` where the elements correspond to the following:
    | Num | Observation                                      | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ------------------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | z-coordinate of the top (height of hopper)       | -Inf | Inf | rootz (torso)                    | slide | position (m)             |
    | 1   | angle of the top                                 | -Inf | Inf | rooty (torso)                    | hinge | angle (rad)              |
    | 2   | angle of the thigh joint                         | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
    | 3   | angle of the leg joint                           | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
    | 4   | angle of the foot joint                          | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
    | 5   | angle of the left thigh joint                    | -Inf | Inf | thigh_left_joint                 | hinge | angle (rad)              |
    | 6   | angle of the left leg joint                      | -Inf | Inf | leg_left_joint                   | hinge | angle (rad)              |
    | 7   | angle of the left foot joint                     | -Inf | Inf | foot_left_joint                  | hinge | angle (rad)              |
    | 8   | velocity of the x-coordinate of the top          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
    | 9   | velocity of the z-coordinate (height) of the top | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
    | 10  | angular velocity of the angle of the top         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
    | 11  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
    | 12  | angular velocity of the leg hinge                | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
    | 13  | angular velocity of the foot hinge               | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
    | 14  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_left_joint                 | hinge | angular velocity (rad/s) |
    | 15  | angular velocity of the leg hinge                | -Inf | Inf | leg_left_joint                   | hinge | angular velocity (rad/s) |
    | 16  | angular velocity of the foot hinge               | -Inf | Inf | foot_left_joint                  | hinge | angular velocity (rad/s) |
    ### Rewards
    The reward consists of three parts:
    - *healthy_reward*: Every timestep that the walker is alive, it receives a fixed reward of value `healthy_reward`,
    - *forward_reward*: A reward of walking forward which is measured as
    *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*.
    *dt* is the time between actions and is dependeent on the frame_skip parameter
    (default is 4), where the frametime is 0.002 - making the default
    *dt = 4 * 0.002 = 0.008*. This reward would be positive if the walker walks forward (right) desired.
    - *ctrl_cost*: A cost for penalising the walker if it
    takes actions that are too large. It is measured as
    *`ctrl_cost_weight` * sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is
    a parameter set for the control and has a default value of 0.001
    The total reward returned is ***reward*** *=* *healthy_reward bonus + forward_reward - ctrl_cost* and `info` will also contain the individual reward terms
    ### Starting State
    All observations start in state
    (0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with a uniform noise in the range of [-`reset_noise_scale`, `reset_noise_scale`] added to the values for stochasticity.
    ### Episode End
    The walker is said to be unhealthy if any of the following happens:
    1. Any of the state space values is no longer finite
    2. The height of the walker is ***not*** in the closed interval specified by `healthy_z_range`
    3. The absolute value of the angle (`observation[1]` if `exclude_current_positions_from_observation=False`, else `observation[2]`) is ***not*** in the closed interval specified by `healthy_angle_range`
    If `terminate_when_unhealthy=True` is passed during construction (which is the default),
    the episode ends when any of the following happens:
    1. Truncation: The episode duration reaches a 1000 timesteps
    2. Termination: The walker is unhealthy
    If `terminate_when_unhealthy=False` is passed, the episode is ended only when 1000 timesteps are exceeded.
    ### Arguments
    No additional arguments are currently supported in v2 and lower.
    ```
    env = gym.make('Walker2d-v4')
    ```
    v3 and beyond take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.
    ```
    env = gym.make('Walker2d-v4', ctrl_cost_weight=0.1, ....)
    ```
    | Parameter                                    | Type      | Default          | Description                                                                                                                                                       |
    | -------------------------------------------- | --------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"walker2d.xml"` | Path to a MuJoCo model                                                                                                                                            |
    | `forward_reward_weight`                      | **float** | `1.0`            | Weight for _forward_reward_ term (see section on reward)                                                                                                          |
    | `ctrl_cost_weight`                           | **float** | `1e-3`           | Weight for _ctr_cost_ term (see section on reward)                                                                                                                |
    | `healthy_reward`                             | **float** | `1.0`            | Constant reward given if the ant is "healthy" after timestep                                                                                                      |
    | `terminate_when_unhealthy`                   | **bool**  | `True`           | If true, issue a done signal if the z-coordinate of the walker is no longer healthy                                                                               |
    | `healthy_z_range`                            | **tuple** | `(0.8, 2)`       | The z-coordinate of the top of the walker must be in this range to be considered healthy                                                                          |
    | `healthy_angle_range`                        | **tuple** | `(-1, 1)`        | The angle must be in this range to be considered healthy                                                                                                          |
    | `reset_noise_scale`                          | **float** | `5e-3`           | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                    |
    | `exclude_current_positions_from_observation` | **bool**  | `True`           | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |
    ### Version History
    * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
    * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco_py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)
    """
'''
class Walker2dReflect(AugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__()

        self.mask_right = np.zeros(17, dtype=bool)
        self.mask_right[2:4+1] = 1
        self.mask_right[11:13+1] = 1

        self.mask_left = np.zeros(17, dtype=bool)
        self.mask_left[5:7+1] = 1
        self.mask_left[14:16+1] = 1

    def reflect_obs(self, obs):
        r = obs[:, self.mask_right].copy()
        l = obs[:, self.mask_left].copy()
        obs[:, self.mask_left] = r
        obs[:, self.mask_right] = l

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                p=None
                ):

        self.reflect_obs(obs)
        self.reflect_obs(next_obs)

        ra = action[:, :3].copy()
        la = action[:, 3:].copy()
        action[:, :3] = la
        action[:, 3:] = ra

        return obs, next_obs, action, reward, done, infos


def tmp():
    env = gym.make('Walker2d-v4', render_mode='human')
    env.reset()

    f = Walker2dReflect()

    qpos_orginal = env.data.qpos.copy()
    qvel_original = env.data.qvel.copy()
    # env.render()
    # time.sleep(2)

    for i in range(9):
        qpos = qpos_orginal.copy()
        qvel = qvel_original.copy()
        # qpos[2:] = 0
        # qvel[:] = 0
        # # qpos[2:5] = -np.pi / 4 * (-1)**i
        # # qpos[6:9] = +np.pi / 4 * (-1)**i
        # qpos[3] = 0.2  *(-1)**i
        # # qpos[4] = 0.4  *(-1)**i
        # # qpos[5] = 0.6  *(-1)**i
        #
        # qpos[6] = -0.2  *(-1)**i
        # qpos[7] = -0.4  *(-1)**i
        # qpos[8] = -0.6  *(-1)**i
        print(qpos)
        print(qvel)
        env.set_state(qpos, qvel)
        print(env.get_obs())

        action = np.zeros(6)
        if i % 2 == 0:
            action[5] = -1
        else:
            action[5] = -1
        for j in range(20):
            env.step(action)
        print(env.get_obs())
        print()
        env.render()
        time.sleep(3)

    # for i in range(10):
    #     # qpos = np.random.uniform(-np.pi, np.pi, size=qpos_orginal.shape)
    #     qpos = qpos_orginal.copy()
    #     qvel = qvel_original.copy()
    #     qpos[3] = np.pi / 4
    #     qpos[4] = np.pi / 4
    #     # qvel = np.random.uniform(-1, 1, size=qpos_orginal.shape)
    #     env.set_state(qpos, qvel)
    #     env.render()
    #     time.sleep(1)
    #
    #     # augment
    #     obs = env.get_obs()
    #     obs_aug = f.reflect_obs(obs)
    #     qpos_aug = qpos.copy()
    #     qpos_aug[1:] = obs[0, :8]
    #     qvel = obs[0, 8:]
    #
    #     env.set_state(qpos_aug, qvel)
    #     env.render()
    #     time.sleep(2)

def check_valid(env, aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info):

    # set env to aug_obs
    # env = gym.make('Walker2d-v4', render_mode='human')

    # env.reset()
    qpos, qvel = env.obs_to_q(aug_obs)
    x_pos = np.array([0])
    qpos = np.concatenate([x_pos, qpos])
    env.set_state(qpos, qvel)

    # determine ture next_obs, reward
    next_obs_true, reward_true, terminated_true, truncated_true, info_true = env.step(aug_action)
    print(aug_next_obs - next_obs_true)
    assert np.allclose(aug_next_obs, next_obs_true)
    assert np.allclose(aug_reward, reward_true)

WALKER2D_AUG_FUNCTIONS = {
    'reflect': Walker2dReflect,
}


if __name__ == "__main__":
    import gym
    env = gym.make('Walker2d-v4', render_mode='human')

    tmp()

    observations, next_observations, actions, rewards, dones, infos = simulate(
        model=None, env=env, num_episodes=1, seed=np.random.randint(1,1000000), render=False, flatten=True, verbose=0)

    observations = np.expand_dims(observations, axis=1)
    next_observations = np.expand_dims(next_observations, axis=1)
    actions = np.expand_dims(actions, axis=1)
    rewards = np.expand_dims(rewards, axis=1)
    dones = np.expand_dims(dones, axis=1)
    infos = np.expand_dims(infos, axis=1)

    aug_func = Walker2dReflect()
    aug_n = 1

    for j in range(observations.shape[0]):
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info = aug_func.augment(
            aug_n, observations[j], next_observations[j], actions[j], rewards[j], dones[j], infos[j])
        for k in range(aug_n):
            check_valid(env, aug_obs[k], aug_next_obs[k], aug_action[k], aug_reward[k], aug_done[k], aug_info[k])