import torch
import treetensor.numpy as tnp
import treetensor.torch as ttorch

from copy import deepcopy

from ding.data import DequeBuffer
from ding.framework.middleware import offpolicy_data_fetcher
from ding.reward_model import RndRewardModel


from garage import Trainer


class RNDTrainer(Trainer):
    def __init__(self, snapshot_config, rnd_config, obs_space):
        self._step_itr = None
        super().__init__(snapshot_config)

        self._reward_model = self._setup_rnd_model(rnd_config.model, obs_space)
        self._rnd_buffer = DequeBuffer(size=rnd_config.buffer.size)
        self._rnd_data_fetcher = self._setup_rnd_data_fetcher(rnd_config)
        self._current_episodes = None

    def _setup_rnd_model(self, rnd_config, obs_space):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rnd_config = rnd_config.copy()

        if len(obs_space.shape) > 1:
            raise RuntimeError('Cannot initialize rnd model with non-vector-like observation_space.')

        rnd_config.obs_shape = obs_space.shape[0]

        return RndRewardModel(rnd_config, device)

    def _setup_rnd_data_fetcher(self, config):
        return offpolicy_data_fetcher(config, self._rnd_buffer)

    def obtain_episodes(self,
                        itr,
                        batch_size=None,
                        agent_update=None,
                        env_update=None):
        episodes = super().obtain_episodes(itr, batch_size=batch_size, agent_update=agent_update, env_update=env_update)
        self._current_episodes = deepcopy(episodes)
        episodes = self._append_intrinsic_rewards(episodes)
        return episodes

    def _append_intrinsic_rewards(self, episodes):
        # TODO need to add to padded rewards too
        # TODO append intrinsic rewards
        # TODO need to make sure the extrinsic rewards are somehow normalized here
        # TODO tensorboard writer

        timesteps = [
            ttorch.tensor({'obs': obs, 'reward': reward, 'done': done}).float()
            for obs, reward, done in zip(episodes.observations, episodes.rewards, episodes.terminals)
        ]

        self._reward_model.collect_data(timesteps)
        augmented_rewards = self._reward_model.estimate(timesteps)
        episodes.rewards[:] = [x['reward'].item() for x in augmented_rewards]

        return episodes

    def _train_rnd_model(self):
        self._reward_model.train()
        self._reward_model.clear_data()
        return None

    @property
    def step_itr(self):
        return self._step_itr

    @step_itr.setter
    def step_itr(self, value):
        if self._step_itr is not None:
            self._train_rnd_model()
        self._step_itr = value
