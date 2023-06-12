import torch
import numpy as np

from copy import deepcopy

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional

from garage import _Default, make_optimizer
from garage.torch.algos import DDPG, PPO
from garage.torch.modules import MLPModule
from garage.torch._functions import zero_optim_grads


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
        episodes = self.append_intrinsic_rewards(episodes)
        super().train_once(itr, episodes)
        self._rnd_model.train_once(episodes.observations)
        # Train rnd model
        # Log intrinsic reward training

    def append_intrinsic_rewards(self, episodes):
        episodes = deepcopy(episodes)
        intrinsic_rewards = self._rnd_model.compute_intrinsic_rewards(episodes.observations)
        episodes.rewards += intrinsic_rewards
        # TODO log the intrinsic/extrinsic rewards
        return episodes


class RNDModel:
    def __init__(self,
                 obs_dim,
                 output_dim=128,
                 hidden_sizes=(64, 64),
                 batch_size=64,
                 n_train_steps=32,
                 intrinsic_reward_weight=0.01,
                 extrinsic_reward_norm=True,
                 extrinsic_reward_norm_max=1,
                 predictor_optimizer=torch.optim.Adam,
                 predictor_lr=1e-3):

        self.batch_size = batch_size
        self.n_train_steps = n_train_steps
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.extrinsic_reward_norm = extrinsic_reward_norm
        self.extrinsic_reward_norm_max = extrinsic_reward_norm_max

        self._reward_model = RNDNetwork(obs_dim, output_dim, hidden_sizes)
        self._reward_model_optimizer = make_optimizer(predictor_optimizer,
                                                      module=self._reward_model.predictor,
                                                      lr=predictor_lr)

    def train_once(self, observations):
        dataset = RNDDataset(observations)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        dataloader_iter = iter(dataloader)

        losses = []

        for _ in range(self.n_train_steps):
            loss = self._train_step(obs_minibatch=next(dataloader_iter))
            loss.append(losses)

        # TODO log this

        return losses

    def _train_step(self, obs_minibatch):
        predicted_feature, target_feature = self._reward_model(obs_minibatch)
        mse_loss = nn.MSELoss()
        loss = mse_loss(predicted_feature, target_feature.detach())
        zero_optim_grads(self._reward_model_optimizer)
        loss.backward()
        self._reward_model_optimizer.step()
        return loss.detach()

    def compute_intrinsic_rewards(self, obs):
        with torch.no_grad():
            predicted_feature, target_feature = self._reward_model(obs)
            mse_f = nn.MSELoss(reduction='none')
            mse = mse_f(predicted_feature, target_feature).mean(dim=1)
            # TODO do we want to normalize?
            return mse.detach().numpy()


class RNDNetwork(nn.Module):
    def __init__(self, obs_dim, output_dim, hidden_sizes):
        super().__init__()
        self.target = MLPModule(obs_dim, output_dim, hidden_sizes)
        self.predictor = MLPModule(obs_dim, output_dim, hidden_sizes)

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor):
        predicted_feature = self.predictor(obs)
        with torch.no_grad():
            target_feature = self.target(obs)
        return predicted_feature, target_feature


class RNDDataset(torch.utils.data.Dataset):
    def __init__(self, obs):
        super().__init__()
        self.obs = obs

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, index):
        return self.obs[index]
