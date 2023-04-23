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

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_batch()
