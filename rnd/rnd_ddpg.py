import torch
import numpy as np

from copy import deepcopy

from garage import _Default
from garage.torch.algos import DDPG


class RNDDDPG(DDPG):
    def __init__(
            self,
            env_spec,
            policy,
            qf,
            replay_buffer,
            sampler,
            rnd_model,
            *,
            steps_per_epoch=20,
            n_train_steps=50,
            max_episode_length_eval=None,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            exploration_policy=None,
            target_update_tau=0.01,
            discount=0.99,
            policy_weight_decay=0,
            qf_weight_decay=0,
            policy_optimizer=torch.optim.Adam,
            qf_optimizer=torch.optim.Adam,
            policy_lr=_Default(1e-4),
            qf_lr=_Default(1e-3),
            clip_pos_returns=False,
            clip_return=np.inf,
            max_action=None,
            reward_scale=1.
    ):
        super().__init__(
            env_spec,
            policy,
            qf,
            replay_buffer,
            sampler,
            steps_per_epoch=steps_per_epoch,
            n_train_steps=n_train_steps,
            max_episode_length_eval=max_episode_length_eval,
            buffer_batch_size=buffer_batch_size,
            min_buffer_size=min_buffer_size,
            exploration_policy=exploration_policy,
            target_update_tau=target_update_tau,
            discount=discount,
            policy_weight_decay=policy_weight_decay,
            qf_weight_decay=qf_weight_decay,
            policy_optimizer=policy_optimizer,
            qf_optimizer=qf_optimizer,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            clip_pos_returns=clip_pos_returns,
            clip_return=clip_return,
            max_action=max_action,
            reward_scale=reward_scale
        )
        # TODO define this
        self._rnd_model = rnd_model

    def train_once(self, itr, episodes):
        # Add intrinsic rewards to episodes
        # Log intrinsic reward info
        episodes = self.transform_rewards(episodes)
        super().train_once(itr, episodes)
        self._rnd_model.train_once(episodes.observations)
        # Train rnd model
        # Log intrinsic reward training

    def transform_rewards(self, episodes):
        episodes = deepcopy(episodes)
        transformed_rewards = self._rnd_model.transform_rewards(episodes.observations, episodes.rewards)
        episodes.rewards[:] = transformed_rewards
        return episodes
