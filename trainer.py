import expfig
import logging
import pandas as pd
import os
import torch
import json
import warnings

from pathlib import Path
from abc import abstractmethod
from copy import deepcopy
from typing import Union

from garage.torch.algos.dqn import DQN
from garage.torch.algos.ddpg import DDPG
from garage.torch.algos.ppo import PPO
from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler, RaySampler
from garage.trainer import Trainer as GarageTrainer
from garage.np.exploration_policies import EpsilonGreedyPolicy, AddOrnsteinUhlenbeckNoise
from garage.torch.policies import DiscreteQFArgmaxPolicy, DeterministicMLPPolicy, GaussianMLPPolicy
from garage.torch.q_functions import DiscreteMLPQFunction, ContinuousMLPQFunction
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.replay_buffer import PathBuffer

from callback import GarageCallback
from envs import GymEnv
from microgrid_loader import microgrid_from_config

import pymgrid

from pymgrid.algos import ModelPredictiveControl, RuleBasedControl
from pymgrid.envs import ContinuousMicrogridEnv, DiscreteMicrogridEnv


DEFAULT_CONFIG = Path(__file__).parent / 'config/default_config.yaml'

pymgrid.add_pymgrid_yaml_representers()


class Trainer:
    algo_name: str
    config: expfig.Config

    def __new__(cls: type, config=None, default=DEFAULT_CONFIG, *args, **kwargs):
        logging.getLogger(__name__).setLevel(logging.INFO)

        config = expfig.Config(config=config, default=default)
        algo = config.algo.type

        if issubclass(cls, (MPCTrainer, RBCTrainer, DQNTrainer, DDPGTrainer, PPOTrainer)):
            config.algo.type = cls.algo_name
        elif isinstance(cls, RLTrainer):
            raise TypeError("Initiating 'RLTrainer' directly is deprecated. Use 'dqn', 'ddpg' or 'ppo' accordingly, "
                            "instead.")
        elif algo.lower() == 'rl':
            raise ValueError("algo type 'rl' is deprecated. Use 'dqn', 'ddpg' or 'ppo', accordingly, instead.")
        elif algo.lower() == 'dqn':
            cls = DQNTrainer
        elif algo.lower() == 'ddpg':
            cls = DDPGTrainer
        elif algo.lower() == 'ppo':
            cls = PPOTrainer
        elif algo.lower() == 'mpc':
            cls = MPCTrainer
        elif algo.lower() == 'rbc':
            cls = RBCTrainer
        else:
            raise ValueError(f"Unrecognized algo type '{algo}'.")

        cls.config = config
        return super().__new__(cls)

    def __init__(self, serialize_config=True, *args, **kwargs):
        self.microgrid = self._setup_microgrid()
        self.algo = self._setup_algo()
        self.log_dirs = self._get_log_dir()
        if serialize_config:
            self.serialize_config()

    def _setup_microgrid(self):
        return microgrid_from_config(self.config.microgrid)

    def _get_log_dir(self):
        log_config = self.config.context
        experiment_name = log_config.experiment_name if log_config.experiment_name is not None else ''

        subdirs = ['config', 'train_log', 'evaluate_log']

        if log_config.log_dir.parent is None or log_config.log_dir.parent == 'null':
            log_dir = None
        else:
            log_dir_params = self._get_log_dir_params(log_config.log_dir.from_keys)
            log_dir = os.path.join(
                log_config.log_dir.parent,
                self.algo_name,
                experiment_name,
                *log_dir_params
            )

        try:
            log_dir = expfig.make_sequential_log_dir(
                log_dir,
                subdirs=subdirs,
                use_existing_dir=log_config.log_dir.use_existing_dir
            )
        except OSError as e:
            old_log_dir = log_dir
            log_dir = expfig.make_sequential_log_dir(
                None,
                subdirs=subdirs,
                use_existing_dir=log_config.log_dir.use_existing_dir
            )

            warnings.warn(f"Exception encountered when creating log_dir '{old_log_dir}':\n\t{e.__class__.__name__}: {e}"
                          f"\nLogging to temp dir: \n\t{log_dir}")

        return {
            'log_dir': log_dir,
            **{subdir: os.path.join(log_dir, subdir) for subdir in subdirs}
                }

    def _get_log_dir_params(self, log_dir_param_keys):
        dirs = []

        if log_dir_param_keys:
            for keys in log_dir_param_keys:
                split = tuple(keys.split('.'))

                try:
                    value = self.config[split]
                except KeyError as e:
                    deep_key = [s for j, s in enumerate(split) if e.args[0] not in split[:j]]
                    raise KeyError(f'Missing deep key: {"->".join(deep_key)}')

                if pd.api.types.is_number(value):
                    value = round(value, 3)
                dirs.append(f'{split[-1]}_{value}')

        return dirs

    def serialize_config(self):
        self.config.serialize_to_dir(self.log_dirs["config"], use_existing_dir=True, with_default=True)

    @abstractmethod
    def _setup_algo(self):
        pass

    def train_and_evaluate(self):
        self.train()
        return self.evaluate()

    def train(self):
        set_seed(self.config.context.seed)
        self._set_trajectories(train=True)
        self._train(self.log_dirs['train_log'])

        print(f'Logged results in dir: {os.path.abspath(self.log_dirs["train_log"])}')

    def _train(self, log_dir):
        pass

    def evaluate(self, log_dir=None):
        if log_dir is None:
            log_dir = self.log_dirs['evaluate_log']

        self._set_trajectories(evaluate=True)

        output = self._evaluate()

        self.save_with_metadata(output, log_dir, 'log.csv')
        self.save_with_metadata(output.sum(), log_dir, 'log_total.csv')

        print(f'Logged results in dir:\n\t{os.path.abspath(log_dir)}')
        return output

    def _evaluate(self):
        return self.algo.run(verbose=self.config.verbosity > 0)

    def save_with_metadata(self, table, log_dir, fname):
        log_path = os.path.join(log_dir, fname)
        table.to_csv(log_path)

        metadata = self.get_metadata(table)
        metadata_stream = Path(f'{log_path}.tag')

        with metadata_stream.open('w') as f:
            json.dump(metadata, f)

    @staticmethod
    def get_metadata(table):
        return {
            'index_col': [j for j in range(table.index.nlevels)],
            'header': [j for j in range(table.columns.nlevels)] if isinstance(table, pd.DataFrame) else 0
        }

    def _set_trajectories(self, train=False, evaluate=False):
        self.set_trajectory(self.microgrid, train=train, evaluate=evaluate)

    def set_trajectory(self, microgrid, train=False, evaluate=False):
        if train + evaluate != 1:
            raise RuntimeError('One of train and evaluate must be true. Both cannot be true.')

        if train:
            trajectory = self.config.microgrid.trajectory.train
        else:
            trajectory = self.config.microgrid.trajectory.evaluate

        for attr, value in trajectory.items():
            if not hasattr(microgrid, attr):
                raise ValueError(f"Microgrid does not have attribute '{attr}' and it cannot be set.")
            setattr(microgrid, attr, value)

        return microgrid

    @classmethod
    def load(cls, log_dir, additional_config=None, additional_garage_data=None):
        """
        Load a previously trained trainer.
        """
        log_dir = Path(log_dir)
        if not log_dir.exists():
            raise FileNotFoundError(f'Cannot locate dir:\n\t{log_dir.absolute()}')

        default = log_dir / 'config/config_default.yaml'
        config = [log_dir / 'config/config.yaml']

        if additional_config is not None:
            config.append(additional_config)

        instance = cls(config=config, default=default, serialize_config=False)
        garage_data = instance.load_additional_data(log_dir, additional_garage_data)

        if garage_data:
            return instance, garage_data

        return instance

    def load_additional_data(self, log_dir, additional_garage_data):
        pass


class RLTrainer(Trainer):
    algo_name = 'rl'
    env: GymEnv
    eval_env: GymEnv
    env_class: Union[ContinuousMicrogridEnv, DiscreteMicrogridEnv]

    def _setup_algo(self):
        self.warn_custom_params()
        self.env, self.eval_env = self._setup_env()
        algo, self.sampler = self.setup_rl_algo()
        return algo

    def _setup_env(self):
        env = self.env_class.from_microgrid(self.microgrid, observation_keys=self.config.env.observation_keys)
        env = GymEnv(env, max_episode_length=len(env))
        env = self.set_trajectory(env, train=True)
        eval_env = self.set_trajectory(deepcopy(env), evaluate=True)

        return env, eval_env

    def _set_trajectories(self, train=False, evaluate=False):
        super()._set_trajectories(train=train, evaluate=evaluate)
        self.set_trajectory(self.env, train=train, evaluate=evaluate)

    @abstractmethod
    def setup_rl_algo(self):
        pass

    def _setup_sampler(self, exploration_policy):
        sampler_config = self.config.algo.sampler

        sampler_kwargs = {
            'agents': exploration_policy,
            'envs': self.env,
            'max_episode_length': self.env.spec.max_episode_length,
            'n_workers': sampler_config.n_workers,
            'is_tf_worker': False
        }

        if sampler_config.type == 'ray':
            return RaySampler(**sampler_kwargs)
        elif sampler_config.type == 'local':
            return LocalSampler(**sampler_kwargs)
        else:
            raise ValueError(f"Invalid sampler config type {sampler_config.type}, must be 'local' or 'ray'.")

    def warn_custom_params(self):
        algos = ['dqn', 'ddpg', 'ppo']
        algos.remove(self.algo_name)
        for other_algo in algos:
            custom_in_other = self.config.algo[other_algo].symmetric_difference(
                self.config.default_config.algo[other_algo])

            if custom_in_other:
                warnings.warn(f"Custom parameters for algorithm '{other_algo}' will be ignored with "
                              f"algo='{self.algo_name}': \n{custom_in_other.pprint(indent=4)}")

    def _train(self, log_dir):
        if self.config.algo.package == 'garage':
            return self._train_garage(log_dir)
        else:
            raise ValueError(self.config.algo.package)

    def _train_garage(self, log_dir):
        log_config = self.config.context
        train_config = self.config.algo.train

        @wrap_experiment(name=log_dir,
                         snapshot_mode='gap_overwrite',
                         snapshot_gap=log_config.snapshot_gap,
                         archive_launch_repo=False,
                         log_dir=log_dir,
                         use_existing_dir=True)
        def train(ctxt=None):
            garage_trainer = GarageTrainer(ctxt)

            garage_trainer.setup(self.algo, self.env)
            garage_trainer.train(n_epochs=train_config.n_epochs, batch_size=train_config.batch_size)

            self.env.close()

            print(f'Logged results in dir: {log_dir}')

        train()

    def _evaluate(self):
        env = self.eval_env.unwrapped
        obs = env.reset()
        done = False

        while not done:
            obs, reward, done, _ = env.step(self.algo.policy.get_action(obs)[0])

        return env.log

    def load_additional_data(self, log_dir, additional_garage_data=None):
        from garage.experiment import Snapshotter

        garage_data = Snapshotter().load(Path(log_dir) / 'train_log')
        self.algo = garage_data['algo']
        self.env = garage_data['env']

        if additional_garage_data is not None:
            if isinstance(additional_garage_data, str):
                return garage_data[additional_garage_data]

            assert pd.api.types.is_list_like(additional_garage_data), 'additional_garage_data should be a key (str)' \
                                                                      'or list of keys (list-like of str).'
            return {k: garage_data[k] for k in additional_garage_data}


class DQNTrainer(RLTrainer):
    algo_name = 'dqn'
    env_class = DiscreteMicrogridEnv
    eval_env: DiscreteMicrogridEnv

    def _setup_env(self):
        env, eval_env = super()._setup_env()
        self._set_callback(eval_env)

        return env, eval_env

    def _set_callback(self, eval_env):
        if self.config.algo.package == 'garage':
            num_eval_episodes = self.config.algo.dqn.get('num_eval_episodes', 10)
            max_episode_length = self.config.algo.dqn.get('max_episode_length_eval') or eval_env.spec.max_episode_length

            callback = GarageCallback(num_eval_episodes, max_episode_length)

            eval_env.step_callback = callback.step
            eval_env.reset_callback = callback.reset
        else:
            raise ValueError(self.config.algo.package)

    def _set_trajectories(self, train=False, evaluate=False):
        super()._set_trajectories(train=train, evaluate=evaluate)
        self.set_trajectory(self.eval_env, train=False, evaluate=True)

    def _setup_policies(self):
        train_config = self.config.algo.train

        total_timesteps = train_config.n_epochs * train_config.steps_per_epoch * train_config.batch_size

        qf = DiscreteMLPQFunction(env_spec=self.env.spec, hidden_sizes=self.config.algo.policy.hidden_sizes)
        policy = DiscreteQFArgmaxPolicy(env_spec=self.env.spec, qf=qf)

        exploration_policy = EpsilonGreedyPolicy(env_spec=self.env.spec,
                                                 policy=policy,
                                                 total_timesteps=total_timesteps,
                                                 min_epsilon=self.config.algo.dqn.policy.exploration.min_epsilon,
                                                 max_epsilon=self.config.algo.dqn.policy.exploration.max_epsilon,
                                                 decay_ratio=self.config.algo.dqn.policy.exploration.decay_ratio)
        return qf, policy, exploration_policy

    def setup_rl_algo(self):
        qf, policy, exploration_policy = self._setup_policies()
        sampler = self._setup_sampler(exploration_policy)
        return self._setup_rl_algo(qf, policy, exploration_policy, sampler), sampler

    def _setup_rl_algo(self, qf, policy, exploration_policy, sampler):

        replay_buffer = PathBuffer(capacity_in_transitions=self.config.algo.replay_buffer.buffer_size)

        return DQN(
            env_spec=self.env.spec,
            policy=policy,
            qf=qf,
            replay_buffer=replay_buffer,
            sampler=sampler,
            exploration_policy=exploration_policy,
            eval_env=self.eval_env,
            steps_per_epoch=self.config.algo.train.steps_per_epoch,
            **self.config.algo.general_params,
            **self.config.algo.deterministic_params,
            **self.config.algo.dqn.params
        )


class DDPGTrainer(RLTrainer):
    algo_name = 'ddpg'
    env_class = ContinuousMicrogridEnv

    def setup_rl_algo(self):
        qf, policy, exploration_policy = self._setup_policies()
        sampler = self._setup_sampler(exploration_policy)
        return self._setup_rl_algo(qf, policy, exploration_policy, sampler), sampler

    def _setup_policies(self):
        qf = ContinuousMLPQFunction(env_spec=self.env.spec,
                                    hidden_sizes=self.config.algo.policy.hidden_sizes)

        policy = DeterministicMLPPolicy(env_spec=self.env.spec,
                                        hidden_sizes=self.config.algo.policy.hidden_sizes,
                                        output_nonlinearity=torch.tanh)

        exploration_policy = AddOrnsteinUhlenbeckNoise(self.env.spec, policy,
                                                       sigma=self.config.algo.ddpg.policy.exploration.sigma,
                                                       theta=self.config.algo.ddpg.policy.exploration.theta)

        return qf, policy, exploration_policy

    def _setup_rl_algo(self, qf, policy, exploration_policy, sampler):

        replay_buffer = PathBuffer(capacity_in_transitions=self.config.algo.replay_buffer.buffer_size)

        return DDPG(
            env_spec=self.env.spec,
            policy=policy,
            qf=qf,
            replay_buffer=replay_buffer,
            sampler=sampler,
            exploration_policy=exploration_policy,
            steps_per_epoch=self.config.algo.train.steps_per_epoch,
            **self.config.algo.general_params,
            **self.config.algo.deterministic_params,
            **self.config.algo.ddpg.params
        )


class PPOTrainer(RLTrainer):
    algo_name = 'ppo'
    env_class = ContinuousMicrogridEnv

    def setup_rl_algo(self):
        policy = self._setup_policy()
        value_function = self._setup_value_func()
        sampler = self._setup_sampler(policy)

        return self._setup_rl_algo(policy, value_function, sampler), sampler

    def _setup_policy(self):
        return GaussianMLPPolicy(self.env.spec, hidden_sizes=self.config.algo.policy.hidden_sizes)

    def _setup_value_func(self):
        return GaussianMLPValueFunction(env_spec=self.env.spec, hidden_sizes=self.config.algo.policy.hidden_sizes)

    def _setup_rl_algo(self, policy, value_function, sampler):
        return PPO(
            env_spec=self.env.spec,
            policy=policy,
            value_function=value_function,
            sampler=sampler,
            **self.config.algo.general_params,
            **self.config.algo.ppo.params
        )


class MPCTrainer(Trainer):
    algo_name = 'mpc'

    def _setup_algo(self):
        return ModelPredictiveControl(self.microgrid)


class RBCTrainer(Trainer):
    algo_name = 'rbc'

    def _setup_algo(self):
        return RuleBasedControl(self.microgrid)


if __name__ == '__main__':
    Trainer().train_and_evaluate()
