from garage.sampler import WorkerFactory
from garage.experiment.deterministic import get_seed
from garage import log_performance, EpisodeBatch
from garage.trainer import ExperimentStats

from collections import Iterator
from typing import Union

from trainer import RBCTrainer, MPCTrainer

from pymgrid import algos


class Expert(Iterator):
    def __init__(self, expert_type, episodes_per_batch):
        self.trainer = self._get_trainer(expert_type)
        self.episodes_per_batch = episodes_per_batch
        self.env = self.trainer.env
        self.agent = ExpertAgent(algo=self.trainer.algo, env=self.env)
        self.worker = self._get_worker()
        self.stats = None

    def _get_trainer(self, expert_type):
        if expert_type == 'rbc':
            return RBCTrainer()
        elif expert_type == 'mpc':
            return MPCTrainer()

        raise ValueError(f"expert_type must be 'rbc' or 'mpc', not '{expert_type}'.")

    def _get_worker(self):
        worker = WorkerFactory(
            max_episode_length=self.env.spec.max_episode_length,
            seed=get_seed(),
            n_workers=1
        )(0)
        worker.update_env(self.env)
        worker.update_agent(self.agent)
        return worker

    def generate_batch(self):
        batches = [self.worker.rollout() for _ in range(self.episodes_per_batch)]
        samples = EpisodeBatch.concatenate(*batches)
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


class ExpertAgent:
    def __init__(self, algo: Union['algos.ModelPredictiveControl', 'algos.RuleBasedControl'], env):
        self.algo = algo
        self.env = env
        self.algo.microgrid = env

    def get_action(self, obs):
        algo_action = self.algo.get_action()
        converted = self.env.convert_action(algo_action, to_microgrid=False, normalize=True)
        return converted, {}

    def unwrapped(self):
        return self.algo

    def __getattr__(self, item):
        if item == 'algo' or not hasattr(self, 'algo'):
            raise AttributeError(item)

        return getattr(self.algo, item)
