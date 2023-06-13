import torch
import numpy as np

from copy import deepcopy

from garage import _Default
from garage.torch.algos import DDPG, PPO


class RNDMixin:
    def __init__(self, rnd_model, **kwargs):
        super().__init__(**kwargs)
        self._rnd_model = rnd_model

    def train_once(self, itr, episodes):
        # Add intrinsic rewards to episodes
        # Log intrinsic reward info
        episodes = self.transform_rewards(episodes)
        self._rnd_model.train_once(episodes.observations)
        super().train_once(itr, episodes)
        # Train rnd model
        # Log intrinsic reward training

    def transform_rewards(self, episodes):
        episodes = deepcopy(episodes)
        transformed_rewards = self._rnd_model.transform_rewards(episodes.observations, episodes.rewards)
        episodes.rewards[:] = transformed_rewards
        return episodes


class RNDDDPG(RNDMixin, DDPG):
    pass


class RNDPPO(RNDMixin, PPO):
    pass
