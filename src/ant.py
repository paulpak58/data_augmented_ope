import time
# Inspired from Nicholas Corrado's work
# Revised by Paul Pak
# https://github.com/NicholasCorrado/RL-Augment/blob/gym0.26.0/augment/rl/augmentation_functions/ant.py

from typing import Dict, List, Any
import numpy as np
import gym

# from augment.rl.augmentation_functions import validate_augmentation
from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction
from augment.rl.augmentation_functions.validate import validate_augmentation


class AntReflect(AugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obs_permute = np.arange(27)
        # joint angles
        self.obs_permute[5] = 7
        self.obs_permute[6] = 8
        self.obs_permute[7] = 5
        self.obs_permute[8] = 6
        self.obs_permute[9] = 11
        self.obs_permute[10] = 12
        self.obs_permute[11] = 9
        self.obs_permute[12] = 10
        # joint vels
        self.obs_permute[19] = 21
        self.obs_permute[20] = 22
        self.obs_permute[21] = 19
        self.obs_permute[22] = 20
        self.obs_permute[23] = 25
        self.obs_permute[24] = 26
        self.obs_permute[25] = 23
        self.obs_permute[26] = 24

        self.obs_reflect = np.zeros(27, dtype=bool)
        self.obs_reflect[5:12+1] = True
        self.obs_reflect[13] = True
        self.obs_reflect[17] = True
        self.obs_reflect[18] = True
        self.obs_reflect[19:] = True

        self.action_permute = np.arange(8)
        self.action_permute[0] = 6
        self.action_permute[2] = 4
        self.action_permute[4] = 2
        self.action_permute[6] = 0

        self.action_permute[1] = 7 #-
        self.action_permute[3] = 5 #-
        self.action_permute[5] = 3 #-
        self.action_permute[7] = 1 #-

    def _swap_action_left_right(self, action):
        action[:, :] = action[:, self.action_permute]
        action[:, :] *= -1

    def _reflect_orientation(self, obs):
        obs[:, 3] *= -1
        obs[:, 4] *= -1

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                delta = None,
                p=None
                ):

        obs[:, :] = obs[:, self.obs_permute]
        next_obs[:, :] = next_obs[:, self.obs_permute]

        obs[:, self.obs_reflect] *= -1
        next_obs[:, self.obs_reflect] *= -1
        self._reflect_orientation(obs)
        self._reflect_orientation(next_obs)

        self._swap_action_left_right(action)
        reward_forward = infos[0][0]['reward_forward']
        reward[:] += -2*reward_forward

        return obs, next_obs, action, reward, done, infos



class AntRotate(AugmentationFunction):

    def __init__(self, noise_scale=np.pi/4, **kwargs):
        super().__init__(**kwargs)
        self.noise_scale = noise_scale

    def quat_mul(self, quat0, quat1):
        assert quat0.shape == quat1.shape
        assert quat0.shape[-1] == 4

        # mujoco stores quats as (qw, qx, qy, qz)
        w0 = quat0[..., 3]
        x0 = quat0[..., 0]
        y0 = quat0[..., 1]
        z0 = quat0[..., 2]

        w1 = quat1[..., 3]
        x1 = quat1[..., 0]
        y1 = quat1[..., 1]
        z1 = quat1[..., 2]

        w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
        z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
        quat = np.stack([x, y, z, w], axis=-1)

        assert quat.shape == quat0.shape
        return quat

    def _rotate_torso(self, obs, quat_rotate_by):
        quat_curr = obs[0, 1:4+1]
        quat_result = self.quat_mul(quat0=quat_curr, quat1=quat_rotate_by)
        # quat already normalized
        obs[0, 1:4+1] = quat_result

    def _rotate_vel(self, obs, sin, cos):
        x = obs[:, 13].copy()
        y = obs[:, 14].copy()
        obs[:, 13] = x * cos - y * sin
        obs[:, 14] = x * sin + y * cos

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                delta = None,
                p=None
                ):

        assert obs.shape[0] == 1 # for now.
        alpha = np.random.uniform(low=-self.noise_scale, high=+self.noise_scale)
        sin = np.sin(alpha/2)
        cos = np.cos(alpha/2)

        # mujoco stores quats as (qw, qx, qy, qz)
        quat_rotate_by = np.array([sin, 0, 0, cos])

        self._rotate_torso(obs, quat_rotate_by)
        self._rotate_torso(next_obs, quat_rotate_by)

        # Not sure why we need -alpha here...
        sin = np.sin(-alpha)
        cos = np.cos(-alpha)
        self._rotate_vel(obs, sin, cos)
        self._rotate_vel(next_obs, sin, cos)

        vx = infos[0][0]['x_velocity']
        vy = infos[0][0]['y_velocity']
        reward_forward = infos[0][0]['reward_forward']

        reward[:] -= reward_forward
        reward[:] += vx*cos - vy*sin

        return obs, next_obs, action, reward, done, infos


def check_valid(env, aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info):

    # set env to aug_obs
    # env = gym.make('Walker2d-v4', render_mode='human')

    # env.reset()
    qpos, qvel = aug_obs[:12+1], aug_obs[12+1:]
    x = aug_info['x_position']
    y = aug_info['y_position']
    qpos = np.concatenate((np.array([0,0]), qpos))
    env.set_state(qpos, qvel)

    # determine ture next_obs, reward
    next_obs_true, reward_true, terminated_true, truncated_true, info_true = env.step(aug_action)
    print(aug_next_obs[13:14+1])
    print(next_obs_true[13:14+1])
    print(aug_next_obs - next_obs_true)
    print('here', aug_reward-reward_true)
    print(aug_reward, aug_info)
    print(reward_true, info_true)
    assert np.allclose(aug_next_obs, next_obs_true)
    assert np.allclose(aug_reward, reward_true)

def sanity_check():
    env = gym.make('Ant-v4', render_mode='human', reset_noise_scale=0)
    env.reset()

    f = AntRotate()

    qpos_og = env.data.qpos.copy()
    qvel_og = env.data.qvel.copy()
    qpos = qpos_og.copy()
    qvel = qvel_og.copy()

    alpha = (np.pi+np.pi/4)
    qpos[3] = 1*np.sin(alpha/2)
    qpos[6] = np.cos(alpha/2)
    env.set_state(qpos, qvel)
    print('quat', env.data.qpos[3:6+1])

    action = np.zeros(8)
    # action[0] = -1
    # action[0] = 0.5
    # action[2] = -1


    for j in range(100):
        next_obs, _, _, _, _ = env.step(action)

    true = next_obs.copy()

    qpos = qpos_og.copy()
    qvel = qvel_og.copy()
    alpha = np.pi
    qpos[3] = 1*np.sin(alpha/2)
    qpos[6] = np.cos(alpha/2)
    env.set_state(qpos, qvel)
    print('quat', env.data.qpos[3:6+1])

    action = np.zeros(8)
    # action[0] = 0.5
    # action[2] = -1
    # action[6] = -0.5
    # action[4] = 1

    for j in range(100):
        next_obs, reward, terminated, truncated, info = env.step(action)

    # print('a2')
    # for i in range(27):
    #     print(f'{i}\t{true[i]:.8f}\t{next_obs[i]:.8f}')
    # print()

    obs = next_obs.copy().reshape(1,-1)
    next_obs = next_obs.reshape(1,-1)
    action = action.reshape(1,-1)
    infos = np.array([info])
    rewards = np.array([reward])
    aug_obs, aug_next_obs, x,x,x,x = f._augment(obs, next_obs, action, rewards, 0, infos)
    print('quat', aug_obs[0, 1:4+1])

    print(aug_next_obs[0])

    print()
    print(aug_next_obs[0]-true)
    is_close = np.isclose(aug_next_obs[0],true)
    # print(aug_next_obs[:, ~is_close])
    # sin = np.sin(alpha/2)
    # cos = np.cos(alpha/2)
    # quat_curr = env.data.qpos[3:6+1]
    # quat_rotate = np.array([sin, 0, 0, cos])
    # quat_result = f.quat_mul(quat0=quat_curr, quat1=quat_rotate)
    # print('res', quat_result)

ANT_AUG_FUNCTIONS = {
    'reflect': AntReflect,
    'rotate': AntRotate,
}



if __name__ == "__main__":
    # sanity_check()
    '''
    '''
    env = gym.make('Ant-v4', reset_noise_scale=0)
    aug_func = AntRotate()
    validate_augmentation(env, aug_func, check_valid)