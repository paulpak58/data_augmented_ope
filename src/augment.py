# Inspired by https://github.com/Badger-RL/GuidedDataAugmentationForRobotics/tree/main/src/augment
# Revised by Paul Pak
import tensorflow as tf 
import copy
import numpy as np
from augmentation_function import AbstractSimAugmentationFunction

class RotateReflectTranslate(AbstractSimAugmentationFunction):
    '''
    Translate the robot and ball by the same (delta_x, delta_y).
    '''
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _augment(self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ):

        # random robot position
        absolute_obs = self._convert_to_absolute_obs(obs)
        absolute_next_obs = self._convert_to_absolute_obs(next_obs)

        if self.at_goal(absolute_obs[2], absolute_obs[3]):
            return None, None, None, None, None

        aug_absolute_obs = copy.deepcopy(absolute_obs)
        aug_absolute_next_obs = copy.deepcopy(absolute_next_obs)
        aug_action = action.copy()
        aug_done = done.copy()
        # aug_action = action.copy() if isinstance(action, np.ndarray) else action.numpy()
        # aug_done = done.copy() if isinstance(done, np.ndarray) else done.numpy()

        theta = np.random.uniform(-np.pi/4, np.pi/4)

        M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        aug_absolute_obs[:2] = M.dot(aug_absolute_obs[:2].T).T
        aug_absolute_obs[2:4] = M.dot(aug_absolute_obs[2:4].T).T

        robot_angle = aug_absolute_obs[4] + theta
        if robot_angle < 0:
            robot_angle += 2 * np.pi
        aug_absolute_obs[4] += theta

        aug_absolute_next_obs[:2] = M.dot(aug_absolute_next_obs[:2].T).T
        aug_absolute_next_obs[2:4] = M.dot(aug_absolute_next_obs[2:4].T).T

        next_robot_angle = aug_absolute_next_obs[4] + theta
        if next_robot_angle < 0:
            next_robot_angle += 2 * np.pi
        aug_absolute_next_obs[4] += theta

        if np.random.random() < 0.5:
            aug_absolute_obs[1] *= -1
            aug_absolute_next_obs[1] *= -1
            aug_absolute_obs[3] *= -1
            aug_absolute_next_obs[3] *= -1
            aug_absolute_obs[4] *= -1
            aug_absolute_next_obs[4] *= -1

            aug_action[0] *= -1
            aug_action[1] *= 1
            aug_action[2] *= -1

        xmin = np.min([aug_absolute_obs[0], aug_absolute_next_obs[0], aug_absolute_obs[2], aug_absolute_next_obs[2]])
        ymin = np.min([aug_absolute_obs[1], aug_absolute_next_obs[1], aug_absolute_obs[3], aug_absolute_next_obs[3]])
        xmax = np.max([aug_absolute_obs[0], aug_absolute_next_obs[0], aug_absolute_obs[2], aug_absolute_next_obs[2]])
        ymax = np.max([aug_absolute_obs[1], aug_absolute_next_obs[1], aug_absolute_obs[3], aug_absolute_next_obs[3]])

        # Translate bottom left corner of the righter bounding box containing the robot and ball
        new_x = np.random.uniform(-4500, 4500-(xmax-xmin))
        new_y = np.random.uniform(-3000, 3000-(ymax-ymin))

        delta_x = new_x - xmin
        delta_y = new_y - ymin

        absolute_obs[0] += delta_x
        absolute_obs[1] += delta_y
        absolute_obs[2] += delta_x
        absolute_obs[3] += delta_y

        absolute_next_obs[0] += delta_x
        absolute_next_obs[1] += delta_y
        absolute_next_obs[2] += delta_x
        absolute_next_obs[3] += delta_y

        aug_reward, _ = self.calculate_reward(aug_absolute_next_obs)

        aug_obs = self._convert_to_relative_obs(aug_absolute_obs)
        aug_next_obs = self._convert_to_relative_obs(aug_absolute_next_obs)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done