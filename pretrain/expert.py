from garage.sampler import WorkerFactory
from garage.experiment.deterministic import get_seed
from garage import log_performance, EpisodeBatch
from garage.trainer import ExperimentStats

from collections import Iterator
from typing import Union

from trainer import RBCTrainer, MPCTrainer

from pymgrid import algos


class Expert(Iterator):
    def __init__(self, expert_type, episodes_per_batch, n_workers=1, additional_config=None):
        self.trainer = self._get_trainer(expert_type, additional_config)
        self.episodes_per_batch = episodes_per_batch
        self.env = self.trainer.env
        self.agent = ExpertAgent(algo=self.trainer.algo, env=self.env)
        self.n_workers = n_workers
        self.workers = self._get_workers()
        self.stats = None

    def _get_trainer(self, expert_type, additional_config):
        if additional_config is None:
            additional_config = {}

        if expert_type == 'rbc':
            return RBCTrainer(additional_config)
            # return RBCTrainer([additional_config, {'env.observation_keys': 'exclude_forecast'}])
        elif expert_type == 'mpc':
            return MPCTrainer(additional_config)
            # return MPCTrainer([additional_config, {'env.observation_keys': 'exclude_forecast'}])

        raise ValueError(f"expert_type must be 'rbc' or 'mpc', not '{expert_type}'.")

    def _get_workers(self):
        factory = WorkerFactory(
            max_episode_length=self.env.spec.max_episode_length,
            seed=get_seed(),
            n_workers=self.n_workers
        )

        workers = [factory(i) for i in range(self.n_workers)]

        for worker in workers:
            worker.update_env(self.env)
            worker.update_agent(self.agent)

        return workers

    def generate_batch(self):
        n_eps = 0
        batches = []

        while True:
            for worker in self.workers:
                batches.append(worker.rollout())
                n_eps += 1
            if n_eps >= self.episodes_per_batch:
                break

        samples = EpisodeBatch.concatenate(*batches)
        self._log_rollout(samples)
        return samples

    def _log_rollout(self, samples):
        if self.stats is None:
            self.stats = ExperimentStats(total_itr=0, total_env_steps=0, total_epoch=0, last_episode=None)

        self.stats.total_env_steps += sum(samples.lengths)
        log_performance(self.stats, samples, discount=1.0, prefix='Expert')

    def set_stats(self, experiment_stats: ExperimentStats):
        self.stats = experiment_stats

    def __getstate__(self):
        state = self.__dict__.copy()
        state['workers'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.workers = self._get_workers()

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
        assert self.algo.microgrid is self.env
        algo_action = self.algo.get_action()
        converted = self.env.convert_action(algo_action, to_microgrid=False, normalize=True)
        return converted, {}

    def unwrapped(self):
        return self.algo

    def __getattr__(self, item):
        if item == 'algo' or not hasattr(self, 'algo'):
            raise AttributeError(item)

        return getattr(self.algo, item)
