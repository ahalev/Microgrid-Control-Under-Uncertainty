import yaml

from garage.torch.algos.dqn import DQN
from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler, RaySampler
from garage.trainer import Trainer as GarageTrainer
from garage.np.exploration_policies import EpsilonGreedyPolicy
from garage.torch.policies import DiscreteQFArgmaxPolicy
from garage.torch.q_functions import DiscreteMLPQFunction
from garage.replay_buffer import PathBuffer

from config import Config
from envs import GymEnv


class Trainer:
    def __new__(cls, algo='', *args, **kwargs):
        if algo.lower() == 'rl':
            cls = RLTrainer
        elif algo.lower() == 'mpc':
            pass
        elif algo.lower() == 'rb':
            pass
        else:
            raise ValueError(f"Unrecognized algo '{algo}'.")

        return super().__new__(cls, *args, **kwargs)


class RLTrainer(Trainer):
    def __init__(self, algo='rl', config=None):
        assert algo == 'rl'

        self.config = Config(config=config)
        self.env = self._setup_env()
        self.qf, self.policy, self.exploration_policy = self._setup_policies()
        self.sampler = self._setup_sampler()
        self.algo = self._setup_algo()

    def _setup_env(self):
        env_yaml = f'!{self.config.env.cls}\n{yaml.safe_dump(self.config.env.config.data)}'
        try:
            env = yaml.safe_load(env_yaml)
            return GymEnv(env, max_episode_length=len(env))
        except yaml.YAMLError:
            raise yaml.YAMLError(f'Unable to parse environment yaml:\n{env_yaml}')

    def _setup_policies(self):
        policy_config = self.config.algo.policy
        train_config = self.config.algo.train

        total_timesteps = train_config.batch_size * train_config.steps_per_epoch * train_config.batch_size

        qf = DiscreteMLPQFunction(env_spec=self.env.spec, hidden_sizes=policy_config.hidden_sizes)
        policy = DiscreteQFArgmaxPolicy(env_spec=self.env.spec, qf=qf)

        exploration_policy = EpsilonGreedyPolicy(env_spec=self.env.spec,
                                                 policy=policy,
                                                 total_timesteps=total_timesteps,
                                                 min_epsilon=policy_config.exploration.max_epsilon,
                                                 max_epsilon=policy_config.exploration.min_epsilon,
                                                 decay_ratio=policy_config.exploration.decay_ratio)
        return qf, policy, exploration_policy

    def _setup_sampler(self):
        sampler_config = self.config.algo.sampler

        sampler_kwargs = {
            'agents': self.exploration_policy,
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

    def _setup_algo(self):

        replay_buffer = PathBuffer(capacity_in_transitions=self.config.algo.replay_buffer.buffer_size)

        return DQN(
            env_spec=self.env.spec,
            policy=self.policy,
            qf=self.qf,
            replay_buffer=replay_buffer,
            sampler=self.sampler,
            exploration_policy=self.exploration_policy,
            steps_per_epoch=self.config.algo.train.steps_per_epoch,
            **self.config.algo.dqn
        )

    def serialize_config(self, fname):
        with open(fname, 'w') as f:
            self.config.serialize(f)

    def train(self):
        if self.config.algo.package == 'garage':
            return self._train_garage()
        else:
            raise ValueError(self.config.algo.package)

    def _train_garage(self):
        log_config = self.config.context
        train_config = self.config.algo.train

        name = log_config.experiment_name if log_config.experiment_name is not None \
            else self.algo.__class__.__name__.lower()

        log_dir = f'{log_config.log_dir}/{name}'

        @wrap_experiment(name=name,
                         snapshot_mode='gap',
                         snapshot_gap=log_config.snapshot_gap,
                         archive_launch_repo=False,
                         log_dir=log_dir)
        def train(ctxt=None):
            set_seed(log_config.seed)
            garage_trainer = GarageTrainer(ctxt)

            self.serialize_config(f'{garage_trainer._snapshotter.snapshot_dir}/config.yaml')

            garage_trainer.setup(self.algo, self.env)
            garage_trainer.train(n_epochs=train_config.n_epochs, batch_size=train_config.batch_size)

            self.env.close()

            print(f'Logged results in dir: {garage_trainer._snapshotter.snapshot_dir}')

        train()


if __name__ == '__main__':
    Trainer().train()
