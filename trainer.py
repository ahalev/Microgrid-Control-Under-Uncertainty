import expfig
import os
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
from reward_shaping import *

from pymgrid import envs, Microgrid
from pymgrid.algos import ModelPredictiveControl, RuleBasedControl

ENVS = {
    'DiscreteMicrogridEnv': envs.DiscreteMicrogridEnv,
    'ContinuousMicrogridEnv': envs.ContinuousMicrogridEnv
}

DEFAULT_CONFIG = Path(__file__).parent / 'config/default_config.yaml'


class Trainer:
    algo_name: str
    config: expfig.Config

    def __new__(cls: type, config=None, *args, **kwargs):
        config = expfig.Config(config=config, default=DEFAULT_CONFIG)
        algo = config.algo.type

        if issubclass(cls, (RLTrainer, MPCTrainer, RBCTrainer)):
            pass
        elif algo.lower() == 'rl':
            cls = RLTrainer
        elif algo.lower() == 'mpc':
            cls = MPCTrainer
        elif algo.lower() == 'rbc':
            cls = RBCTrainer
        else:
            raise ValueError(f"Unrecognized algo type '{algo}'.")

        config.algo.type = algo
        cls.config = config
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        self.microgrid = self._setup_microgrid()
        self.algo = self._setup_algo()
        self.log_dirs = self._get_log_dir()
        self.serialize_config(f'{self.log_dirs["config"]}/config.yaml')

    def _setup_microgrid(self):
        if isinstance(self.config.microgrid.config, Microgrid):
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
            setattr(microgrid, attr, value)

    def _get_log_dir(self):
        log_config = self.config.context
        experiment_name = log_config.experiment_name if log_config.experiment_name is not None \
            else self.algo.__class__.__name__.lower()

        subdirs = ['config', 'train_log', 'evaluate_log']

        log_dir = expfig.make_sequential_log_dir(f'{log_config.log_dir}/{self.algo_name}/{experiment_name}',
                                                 subdirs=subdirs)

        return {
            'log_dir': log_dir,
            **{subdir: os.path.join(log_dir, subdir) for subdir in subdirs}
                }

    def serialize_config(self, fname):
        with open(fname, 'w') as f:
            self.config.serialize(f)

    @abstractmethod
    def _setup_algo(self):
        pass

    def train_and_evaluate(self):
        self.train()
        return self.evaluate()

    def train(self):
        self._set_trajectories(train=True)
        self._train(self.log_dirs['train_log'])
        print(f'Logged results in dir: {os.path.abspath(self.log_dirs["train_log"])}')

    def _train(self, log_dir):
        pass

    def evaluate(self):
        self._set_trajectories(evaluate=True)
        output = self.algo.run()
        print(f'Logged results in dir: {self.log_dir}')
        return output

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
            microgrid.set_module_attr(attr, value)

        return microgrid


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
        self.set_trajectory(self.env, train=True, evaluate=False)
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
            set_seed(log_config.seed)
            garage_trainer = GarageTrainer(ctxt)

            garage_trainer.setup(self.algo, self.env)
            garage_trainer.train(n_epochs=train_config.n_epochs, batch_size=train_config.batch_size)

            self.env.close()

            print(f'Logged results in dir: {log_dir}')

        train()


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
