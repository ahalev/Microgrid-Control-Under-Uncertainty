import numpy as np

from trainer import RBCTrainer
from garage import TimeStepBatch, EpisodeBatch
from collections import Iterator


class RBCExpert(Iterator):
    def __init__(self):
        self.trainer = RBCTrainer()
        self.env = self.trainer.env
        self.rbc = self.trainer.algo

    def generate_batch(self):
        obs = self.env.reset()
        done = False

        while not done:
            action = self.rbc.get_action(obs)
            obs, reward, done, info = self.env.step(action)

        return None

    def collect_episode(self):
        return EpisodeBatch(env_spec=self.env.spec,
                     episode_infos=episode_infos,
                     observations=np.asarray(observations),
                     last_observations=np.asarray(last_observations),
                     actions=np.asarray(actions),
                     rewards=np.asarray(rewards),
                     step_types=np.asarray(step_types, dtype=StepType),
                     env_infos=dict(env_infos),
                     agent_infos=dict(agent_infos),
                     lengths=np.asarray(lengths, dtype='i'))

    def generate_batches(self):
        """

        Returns
        -------
        Union[TimeStepBatch, EpisodeBatch]
        """
        yield self.generate_batch()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['worker'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.worker = self._get_worker()

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_batch()


class RBCAgent:
    def __init__(self, rbc, env):
        self.rbc = rbc
        self.env = env
        self.rbc.microgrid = env

    def get_action(self, obs):
        rbc_action = self.rbc.get_action()
        converted = self.env.convert_action(rbc_action, to_microgrid=False, normalize=True)
        return converted, {}

    def unwrapped(self):
        return self.rbc

    def __getattr__(self, item):
        return getattr(self.rbc, item)
