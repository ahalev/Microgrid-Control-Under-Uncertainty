from garage.sampler import WorkerFactory
from garage.experiment.deterministic import get_seed
from garage import log_performance
from garage.trainer import ExperimentStats

from collections import Iterator

from trainer import RBCTrainer


class RBCExpert(Iterator):
    def __init__(self):
        self.trainer = RBCTrainer()
        self.env = self.trainer.env
        self.rbc = RBCAgent(rbc=self.trainer.algo, env=self.env)
        self.worker = self._get_worker()
        self.stats = None

    def _get_worker(self):
        worker = WorkerFactory(
            max_episode_length=self.env.spec.max_episode_length,
            seed=get_seed(),
            n_workers=1
        )(0)
        worker.update_env(self.env)
        worker.update_agent(self.rbc)
        return worker

    def generate_batch(self):
        samples = self.worker.rollout()
        self._log_rollout(samples)
        return samples

    def _log_rollout(self, samples):
        self.stats.total_env_steps += sum(samples.lengths)
        log_performance(self.stats, samples, discount=1.0, prefix='Expert')

    def set_stats(self, experiment_stats: ExperimentStats):
        self.stats = experiment_stats

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
