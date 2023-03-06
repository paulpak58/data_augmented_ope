# Inspired from Nicholas Corrado's work
# Revised by Paul Pak
# https://github.com/NicholasCorrado/RL-Augment/blob/gym0.26.0/augment/rl/augmentation_functions/inverted_pendulum.py

from typing import Dict, List, Any
import numpy as np

from augmentation import AugmentationFunction

class InvertedPendulumTranslate(AugmentationFunction):

    def __init__(self,  noise_scale=0.9, **kwargs):
        super().__init__()
        self.noise_scale = noise_scale
        print(locals())

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                p=None,
                ):

        n = obs.shape[0]
        delta = np.random.uniform(low=-self.noise_scale, high=+self.noise_scale, size=(n,))
        delta_x = next_obs[:,0] - obs[:,0]
        obs[:,0] = delta
        next_obs[:,0] = delta_x + delta

        return obs, next_obs, action, reward, done, infos

class InvertedPendulumReflect(AugmentationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _augment(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
            **kwargs,
    ):
        delta_x = next_obs[:,0] - obs[:,0]

        obs[:,1:] *= -1
        next_obs[:,1:] *= -1
        next_obs[:, 0] -= 2*delta_x
        action *= -1

        return obs, next_obs, action, reward, done, infos

class InvertedPendulumTranslateReflect(AugmentationFunction):

    def __init__(self,  noise_scale=0.9, **kwargs):
        super().__init__(**kwargs)
        self.noise_scale = noise_scale
        print(locals())

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                p=None,
                ):

        n = obs.shape[0]
        delta = np.random.uniform(low=-self.noise_scale, high=+self.noise_scale, size=(n,))
        delta_x = next_obs[:,0] - obs[:,0]
        obs[:,0] = delta
        next_obs[:,0] = delta_x + delta

        obs[:,0:] *= -1
        next_obs[:,0:] *= -1
        action *= -1

        return obs, next_obs, action, reward, done, infos

INVERTED_PENDULUM_AUG_FUNCTIONS = {
    'translate': InvertedPendulumTranslate,
    'reflect': InvertedPendulumReflect,
    'translate_reflect': InvertedPendulumTranslateReflect,
}