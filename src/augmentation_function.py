# Inspired by https://github.com/Badger-RL/GuidedDataAugmentationForRobotics/blob/main/src/augment/augmentation_function.py
# Revised by Paul Pak

import numpy as np
import tensorflow as tf

class BaseAugmentationFunction:

    def __init__(self, env=None, **kwargs):
        self.env = env

    def _deepcopy_transition(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ):
        copy_obs = obs.copy()
        copy_next_obs = next_obs.copy()
        copy_action = action.copy()
        copy_reward = reward.copy()
        copy_done = done.copy()
        # copy_obs = tf.identity(obs)
        # copy_next_obs = tf.identity(next_obs)
        # copy_action = tf.identity(action)
        # copy_reward = tf.identity(reward)
        # copy_done = tf.identity(done)
        return copy_obs, copy_next_obs, copy_action, copy_reward, copy_done
    
    def augment(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        **kwargs
    ):
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done = self._deepcopy_transition(obs, next_obs, action, reward, done)
        return self._augment(aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, **kwargs)
    
    def _augment(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        **kwargs
    ):
        raise NotImplementedError
    
class AbstractSimAugmentationFunction(BaseAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.ball_pos_mask = None
        self.robot_pos_mask = None
        self.x_scale = 9000
        self.y_scale = 6000
        self.scale = np.array([self.x_scale, self.y_scale])
        self.goal_x = 4800
        self.goal_y = 0
        self.goal = np.array([self.goal_x, self.goal_y])
        self.displacement_coef = 0.2
        self.max_dist = np.sqrt(self.x_scale**2 + self.y_scale**2)

    def _sample_robot_pos(self, n=1):
        x = np.random.uniform(-3500, 3500)
        y = np.random.uniform(-2500, 2500)
        return np.array([x, y])
    
    def _sample_robot_angle(self, n=1):
        return np.random.uniform(0, 2*np.pi, size=(n,))
    
    def _convert_to_absolute_obs(self, obs):
        target_x = (self.goal_x - obs[2]*self.x_scale)
        target_y = (self.goal_y - obs[3]*self.y_scale)
        robot_x = (target_x - obs[0]*self.x_scale)
        robot_y = (target_y - obs[1]*self.y_scale)
        relative_x = target_x - robot_x
        relative_y = target_y - robot_y
        relative_angle = np.arctan2(relative_y, relative_x)
        if relative_angle < 0:
            relative_angle += 2*np.pi
        relative_angle_minus_robot_angle = np.arctan2(obs[4], obs[5])
        if relative_angle_minus_robot_angle < 0:
            relative_angle_minus_robot_angle += 2*np.pi
        robot_angle = relative_angle - relative_angle_minus_robot_angle
        if robot_angle < 0:
            robot_angle += 2*np.pi
        return np.array([robot_x, robot_y, target_x, target_y, robot_angle])
    
    def _convert_to_relative_obs(self, obs):
        robot_pos = obs[:2]
        target_pos = obs[2:4]
        robot_x = obs[0]
        robot_y = obs[1]
        target_x = obs[2]
        target_y = obs[3]
        relative_x = target_x - robot_x
        relative_y = target_y - robot_y
        relative_angle = np.arctan2(relative_y, relative_x)
        if relative_angle < 0:
            relative_angle += 2*np.pi
        robot_angle = obs[4]
        if robot_angle < 0:
            robot_angle += 2*np.pi
        goal_delta = self.goal - robot_pos
        goal_relative_angle = np.arctan2(goal_delta[1], goal_delta[0])
        if goal_relative_angle < 0:
            goal_relative_angle += 2*np.pi

        return np.concatenate([
            (target_pos - robot_pos) / self.scale,
            (self.goal - target_pos) / self.scale,
            [np.sin(relative_angle - robot_angle),
            np.cos(relative_angle - robot_angle),
            np.sin(goal_relative_angle - robot_angle),
            np.cos(goal_relative_angle - robot_angle),]
        ])

    def calculate_reward(self, absolute_next_obs):
        robot_x = absolute_next_obs[0]
        robot_y = absolute_next_obs[1]
        target_x = absolute_next_obs[2]
        target_y = absolute_next_obs[3]
        robot_angle = absolute_next_obs[4]
        robot_angle = np.degrees(robot_angle) % 360
        at_goal = self.at_goal(target_x, target_y)
        angle_robot_ball = np.arctan2(target_y - robot_y, target_x - robot_x)
        angle_robot_ball = np.degrees(angle_robot_ball) % 360
        is_facing_ball = abs(angle_robot_ball - robot_angle) < 30

        reward = 0
        if is_facing_ball:
            robot_location = np.array([robot_x, robot_y])
            target_location = np.array([target_x, target_y])
            goal_location = np.array([self.goal_x, self.goal_y])
            distance_robot_target = np.linalg.norm(target_location - robot_location)
            distance_target_goal = np.linalg.norm(goal_location - target_location)
            reward_dist_to_ball = 1/distance_robot_target
            reward_dist_to_goal = 1/distance_target_goal
            reward = 0.9*reward_dist_to_goal + 0.1*reward_dist_to_ball
        if at_goal:
            reward += 1
        return reward, at_goal

    def at_goal(self, target_x, target_y):
        at_goal = False
        if target_x > 4500:
            if target_y < 750 and target_y > -750:
                at_goal = True

        return at_goal
