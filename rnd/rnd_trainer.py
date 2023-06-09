import torch

from ding.reward_model import RndRewardModel
from garage import Trainer


class RNDTrainer(Trainer):
    def __init__(self, snapshot_config, rnd_config, obs_space):
        self._step_itr = None
        super().__init__(snapshot_config)
        self._reward_model = self._setup_rnd_model(rnd_config, obs_space)
        self._current_episodes = None

    def _setup_rnd_model(self, rnd_config, obs_space):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rnd_config = rnd_config.copy()

        if len(obs_space.shape) > 1:
            raise RuntimeError('Cannot initialize rnd model with non-vector-like observation_space.')

        rnd_config.obs_shape = obs_space.shape[0]

        return RndRewardModel(rnd_config, device)

    def obtain_episodes(self,
                        itr,
                        batch_size=None,
                        agent_update=None,
                        env_update=None):
        # TODO append intrinsic rewards
        episodes = super().obtain_episodes(itr, batch_size=batch_size, agent_update=agent_update, env_update=env_update)
        self._current_episodes = episodes.copy()
        episodes = self._append_intrinsic_rewards(episodes)
        return episodes

    def _append_intrinsic_rewards(self, episodes):
        pass

    def _train_rnd_model(self):
        pass

    @property
    def step_itr(self):
        return self._step_itr

    @step_itr.setter
    def step_itr(self, value):
        if self._step_itr is not None:
            self._train_rnd_model()
        self._step_itr = value
