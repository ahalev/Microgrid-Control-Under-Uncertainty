import expfig
import logging
import pandas as pd
import os
import json
import yaml

from pathlib import Path
from abc import abstractmethod
from copy import deepcopy

from garage.torch.algos.dqn import DQN
from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler, RaySampler
from garage.trainer import Trainer as GarageTrainer
from garage.np.exploration_policies import EpsilonGreedyPolicy
from garage.torch.policies import DiscreteQFArgmaxPolicy
from garage.torch.q_functions import DiscreteMLPQFunction
from garage.replay_buffer import PathBuffer

from envs import GymEnv

import pymgrid

from pymgrid.algos import ModelPredictiveControl, RuleBasedControl

ENVS = {
    'DiscreteMicrogridEnv': pymgrid.envs.DiscreteMicrogridEnv,
    'ContinuousMicrogridEnv': pymgrid.envs.ContinuousMicrogridEnv
}

DEFAULT_CONFIG = Path(__file__).parent / 'config/default_config.yaml'

pymgrid.add_pymgrid_yaml_representers()


class Trainer:
    algo_name: str
    config: expfig.Config

    def __new__(cls: type, config=None, default=DEFAULT_CONFIG, *args, **kwargs):
        logging.getLogger(__name__).setLevel(logging.INFO)

        config = expfig.Config(config=config, default=default)
        algo = config.algo.type

        if issubclass(cls, (RLTrainer, MPCTrainer, RBCTrainer)):
            config.algo.type = cls.algo_name
        elif algo.lower() == 'rl':
            cls = RLTrainer
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
        if isinstance(self.config.microgrid.config, pymgrid.Microgrid):
            microgrid = self.config.microgrid.config
        else:
            microgrid_yaml = f'!Microgrid\n{yaml.safe_dump(self.config.microgrid.config.data)}'
            try:
                microgrid = yaml.safe_load(microgrid_yaml)
            except yaml.YAMLError:
                raise yaml.YAMLError(f'Unable to parse microgrid yaml:\n{microgrid_yaml}')

        self._post_process_microgrid(microgrid)
        return microgrid

    def _post_process_microgrid(self, microgrid):
        self._call_microgrid_methods(microgrid)
        self._set_microgrid_attributes(microgrid)

    def _call_microgrid_methods(self, microgrid):
        try:
            methods = self.config.microgrid.methods
        except AttributeError:
            return

        for method, method_params in methods.items():
            getattr(microgrid, method)(**method_params)

    def _set_microgrid_attributes(self, microgrid):
        try:
            attributes = self.config.microgrid.attributes
        except AttributeError:
            return

        for attr, value in attributes.items():
            if isinstance(value, str) and value.startswith('!'):
                try:
                    value = yaml.safe_load(value)
                except yaml.YAMLError:
                    value = yaml.safe_load(f'{value} {{}}')

            setattr(microgrid, attr, value)

    def _get_log_dir(self):
        log_config = self.config.context
        experiment_name = log_config.experiment_name if log_config.experiment_name is not None \
            else self.algo.__class__.__name__.lower()

        subdirs = ['config', 'train_log', 'evaluate_log']

        log_dir_params = self._get_log_dir_params(log_config.log_dir.from_keys)

        log_dir = os.path.join(
            log_config.log_dir.parent,
            self.algo_name,
            experiment_name,
            *log_dir_params
        )

        log_dir = expfig.make_sequential_log_dir(
            log_dir,
            subdirs=subdirs,
            use_existing_dir=log_config.log_dir['use_existing_dir']
        )

        return {
            'log_dir': log_dir,
            **{subdir: os.path.join(log_dir, subdir) for subdir in subdirs}
                }

    def _get_log_dir_params(self, log_dir_param_keys):
        dirs = []

        if log_dir_param_keys:
            for keys in log_dir_param_keys:
                split = tuple(keys.split('.'))
                value = self.config[split]
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

    def evaluate(self):
        log_dir = self.log_dirs['evaluate_log']
        self._set_trajectories(evaluate=True)

        output = self._evaluate()

        self.save_with_metadata(output, log_dir, 'log.csv')
        self.save_with_metadata(output.sum(), log_dir, 'log_total.csv')

        print(f'Logged results in dir:\n\t{os.path.abspath(log_dir)}')
        return output

    def _evaluate(self):
        return self.algo.run(verbose=self.config.context.verbose > 0)

    def save_with_metadata(self, table, log_dir, fname):
        log_path = os.path.join(log_dir, fname)
        table.to_csv(log_path)

        metadata = self.get_metadata(table)
        metadata_stream = Path(f'{log_path}.tag').open('w')
        json.dump(metadata, metadata_stream)

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
    def load(cls, log_dir):
        """
        Load a previously trained trainer.
        """
        log_dir = Path(log_dir)
        if not log_dir.exists():
            raise FileNotFoundError(f'Cannot locate dir:\n\t{log_dir.absolute()}')

        config = log_dir / 'config/config.yaml'
        default = log_dir / 'config/config_default.yaml'

        return cls(config=config, default=default, serialize_config=False)


class RLTrainer(Trainer):
    algo_name = 'rl'
    env = None

    def _setup_algo(self):
        self.env, self.eval_env = self._setup_env()
        qf, policy, exploration_policy = self._setup_policies()
        self.sampler = self._setup_sampler(exploration_policy)
        return self._setup_rl_algo(qf, policy, exploration_policy)

    def _setup_env(self):
        env_cls = ENVS[self.config.env.cls]
        env = env_cls(self.microgrid)
        env = GymEnv(env, max_episode_length=len(env))

        env = self.set_trajectory(env, train=True)
        eval_env = self.set_trajectory(deepcopy(env), evaluate=True)

        return env, eval_env

    def _set_trajectories(self, train=False, evaluate=False):
        super()._set_trajectories(train=train, evaluate=evaluate)
        self.set_trajectory(self.env, train=train, evaluate=evaluate)
        self.set_trajectory(self.eval_env, train=False, evaluate=True)

    def _setup_policies(self):
        policy_config = self.config.algo.policy
        train_config = self.config.algo.train

        total_timesteps = train_config.batch_size * train_config.steps_per_epoch * train_config.batch_size

        qf = DiscreteMLPQFunction(env_spec=self.env.spec, hidden_sizes=policy_config.hidden_sizes)
        policy = DiscreteQFArgmaxPolicy(env_spec=self.env.spec, qf=qf)

        exploration_policy = EpsilonGreedyPolicy(env_spec=self.env.spec,
                                                 policy=policy,
                                                 total_timesteps=total_timesteps,
                                                 min_epsilon=policy_config.exploration.min_epsilon,
                                                 max_epsilon=policy_config.exploration.max_epsilon,
                                                 decay_ratio=policy_config.exploration.decay_ratio)
        return qf, policy, exploration_policy

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

    def _setup_rl_algo(self, qf, policy, exploration_policy):

        replay_buffer = PathBuffer(capacity_in_transitions=self.config.algo.replay_buffer.buffer_size)

        return DQN(
            env_spec=self.env.spec,
            policy=policy,
            qf=qf,
            replay_buffer=replay_buffer,
            sampler=self.sampler,
            exploration_policy=exploration_policy,
            eval_env=self.eval_env,
            steps_per_epoch=self.config.algo.train.steps_per_epoch,
            **self.config.algo.dqn
        )

    def _train(self, log_dir):
        if self.config.algo.package == 'garage':
            return self._train_garage(log_dir)
        else:
            raise ValueError(self.config.algo.package)

    def _train_garage(self, log_dir):
        log_config = self.config.context
        train_config = self.config.algo.train

        name = log_config.experiment_name if log_config.experiment_name is not None \
            else self.algo.__class__.__name__.lower()

        @wrap_experiment(name=name,
                         snapshot_mode='gap',
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
        env = self.env.unwrapped
        obs = env.reset()
        done = False

        while not done:
            obs, reward, done, _ = env.step(self.algo.policy.get_action(obs))

        return self.env.log

    @classmethod
    def load(cls, log_dir):
        from garage.experiment import Snapshotter

        garage_data = Snapshotter().load(log_dir / 'train_log')

        instance = super().load(log_dir)
        instance.algo = garage_data['algo']
        instance.env = garage_data['env']

        return instance


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
