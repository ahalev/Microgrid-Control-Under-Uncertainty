import yaml

from abc import abstractmethod

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

from pymgrid import envs
from pymgrid.algos import ModelPredictiveControl, RuleBasedControl

ENVS = {
    'DiscreteMicrogridEnv': envs.DiscreteMicrogridEnv,
    'ContinuousMicrogridEnv': envs.ContinuousMicrogridEnv
}


class Trainer:
    algo_name: str
    config: Config

    def __new__(cls: type, config=None, *args, **kwargs):
        config = Config(config=config)
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

    def _setup_microgrid(self):
        microgrid_yaml = f'!Microgrid\n{yaml.safe_dump(self.config.microgrid.config.data)}'
        try:
            microgrid = yaml.safe_load(microgrid_yaml)
            return microgrid
        except yaml.YAMLError:
            raise yaml.YAMLError(f'Unable to parse microgrid yaml:\n{microgrid_yaml}')

    def _get_log_dir(self, log_dir, experiment_name):
        return f'{log_dir}/{self.algo_name}/{experiment_name}'


    def serialize_config(self, fname):
        with open(fname, 'w') as f:
            self.config.serialize(f)

    def train(self):
        return self.algo.run()

    @abstractmethod
    def _setup_algo(self):
        pass


class RLTrainer(Trainer):
    algo_name = 'rl'
    env = None

    def _setup_algo(self):
        self.env = self._setup_env()
        qf, policy, exploration_policy = self._setup_policies()
        self.sampler = self._setup_sampler(exploration_policy)
        return self._setup_rl_algo(qf, policy, exploration_policy)

    def _setup_env(self):
        env_cls = ENVS[self.config.env.cls]
        env = env_cls(self.microgrid)
        return GymEnv(env, max_episode_length=len(env))

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
            steps_per_epoch=self.config.algo.train.steps_per_epoch,
            **self.config.algo.dqn
        )

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

        log_dir = self._get_log_dir(log_config.log_dir, name)

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


class MPCTrainer(Trainer):
    algo_name = 'mpc'

    def _setup_algo(self):
        return ModelPredictiveControl(self.microgrid)


class RBCTrainer(Trainer):
    algo_name = 'rbc'

    def _setup_algo(self):
        return RuleBasedControl(self.microgrid)


if __name__ == '__main__':
    Trainer().train()