import expfig
import functools
import pandas as pd
import os
import subprocess
import torch
import json
import warnings
import wandb

from pathlib import Path
from abc import abstractmethod
from copy import deepcopy
from dowel import tabular, set_wandb_env_keys
from typing import Union

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed, get_seed
from garage.sampler import LocalSampler, RaySampler
from garage.trainer import Trainer as GarageTrainer
from garage.np.exploration_policies import EpsilonGreedyPolicy, AddOrnsteinUhlenbeckNoise
from garage.torch.algos import DQN, DDPG, PPO
from garage.torch.policies import DiscreteQFArgmaxPolicy, DeterministicMLPPolicy, GaussianMLPPolicy, TanhGaussianMLPPolicy
from garage.torch.q_functions import DiscreteMLPQFunction, ContinuousMLPQFunction
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.replay_buffer import PathBuffer

from callback import GarageCallback
from envs import GymEnv, DomainRandomizationWrapper, ForcedGensetWrapper, DomainRandomizationWrapperV2
from microgrid_loader import microgrid_from_config
from rnd import RNDModel, RNDDDPG, RNDPPO

import pymgrid

from pymgrid.algos import ModelPredictiveControl, RuleBasedControl
from pymgrid.envs import ContinuousMicrogridEnv, DiscreteMicrogridEnv, NetLoadContinuousMicrogridEnv


DEFAULT_CONFIG = Path(__file__).parent / 'config/default_config.yaml'

pymgrid.add_pymgrid_yaml_representers()


class Trainer:
    algo_name: str
    config: expfig.Config
    env_class = ContinuousMicrogridEnv

    def __new__(cls: type, config=None, default=DEFAULT_CONFIG, *args, **kwargs):
        config = expfig.Config(config=config, default=default)
        algo = config.algo.type

        if issubclass(cls, (MPCTrainer, RBCTrainer, DQNTrainer, DDPGTrainer, PPOTrainer, PreTrainer)):
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
        elif algo.lower() == 'pretrain':
            cls = PreTrainer
        else:
            raise ValueError(f"Unrecognized algo type '{algo}'.")

        cls.config = config
        return super().__new__(cls)

    def __init__(self, *args, setup_algo=True, serialize_config=True, **kwargs):
        set_seed(self.config.context.seed)

        self.log_dirs = self._get_log_dir()
        self.logger = expfig.logging.get_logger()

        self.microgrid = self._setup_microgrid()
        self.env, self.eval_env = self._setup_env()
        self.algo = self._setup_algo(setup_algo=setup_algo)
        self.baseline_rewards = self._compute_baselines(self.config.context.wandb.plot_baseline)

        self.has_wandb = self._set_wandb_env_keys()

        self.serialize_config(serialize_config)

    def _setup_microgrid(self):
        return microgrid_from_config(self.config.microgrid)

    def _set_wandb_env_keys(self):
        return False

    def _setup_wandb(self, log_dir=None, **summary_info):
        if not self.has_wandb or log_dir == -1:
            return

        experiment_name, tags = self.experiment_name_tags(clip_tags=True)

        wandb_creator = functools.partial(
            wandb.init,
            project='gridrl',
            name=experiment_name,
            group=self.config.context.wandb.group,
            config=self.config.to_dict(),
            tags=tags,
        )

        def _wandb_creator(*args, **kwargs):
            s_info = {**summary_info, 'log_dir': kwargs.get('dir')}
            wandb = wandb_creator(*args, **kwargs)
            wandb.summary.update(s_info)

            for artifact in self._wandb_artifacts():
                wandb.log_artifact(artifact)

            return wandb

        if log_dir is not None:
            return _wandb_creator(dir=log_dir)

        return _wandb_creator

    def _wandb_artifacts(self):
        config_artifact = wandb.Artifact(
            name='config',
            type='configs',
            description='Config, default config and difference between the two.'
        )

        config_artifact.add_dir(self.log_dirs['config'])

        return [config_artifact]

    def _setup_env(self):
        env_kwargs = self._pre_env_setup()

        valid_obs_keys = self.microgrid.state_series().index.get_level_values(-1).unique().union(['net_load'])
        overlap = valid_obs_keys.intersection(self.config.env.observation_keys).tolist()
        invalid_obs_keys = pd.Index(self.config.env.observation_keys).difference(overlap)

        if not invalid_obs_keys.empty:
            self.logger.warning(f'The following observation keys are invalid for the current microgrid, will be '
                                f'ignored:\n\t{invalid_obs_keys.tolist()}\n\t'
                                f'Retaining these observation keys:{overlap}')

        env = self.env_class.from_microgrid(self.microgrid,
                                            observation_keys=overlap,
                                            **env_kwargs)

        env, eval_env = self.wrap_env(env)

        env = self.set_trajectory(env, train=True)
        eval_env = self.set_trajectory(eval_env, evaluate=True)

        return env, eval_env

    def wrap_env(self, env, return_copy=True):
        # TODO (ahalev) do this better
        max_episode_length = getattr(
            self.config.microgrid.trajectory.train.trajectory_func, 'trajectory_length', len(env)
        )

        env = GymEnv(env, max_episode_length=max_episode_length)

        if return_copy:
            return env, deepcopy(env)

        return env

    def experiment_name_tags(self, clip_tags=False):
        tags = [self.algo_name]

        if self.config.context.experiment_name:
            tags.append(self.config.context.experiment_name)

        tags.extend(self._get_log_dir_params(self.config.context.log_dir.from_keys))

        experiment_name = self.config.context.experiment_name or '-'.join(tags)

        if clip_tags:
            clipped = []
            for tag in tags:
                if len(tag) > 64:
                    clipped_tag = tag[-64:]
                    self.logger.warning(f"tag '{tag}' too long, clipped to '{clipped_tag}'")
                    clipped.append(clipped_tag)
                else:
                    clipped.append(tag)

            return experiment_name, clipped

        return experiment_name, tags

    def _get_log_dir(self):
        log_config = self.config.context

        if log_config.disable_logging:
            return dict()

        subdirs = ['config', 'train_log', 'evaluate_log']

        if log_config.log_dir.parent is None or log_config.log_dir.parent == 'null':
            log_dir = None
        else:
            _, tags = self.experiment_name_tags()
            log_dir = os.path.join(log_config.log_dir.parent, *tags)

        try:
            log_dir = expfig.make_sequential_log_dir(
                log_dir,
                subdirs=subdirs,
                use_existing_dir=log_config.log_dir.use_existing_dir,
                logger_file='output.log'
            )
        except OSError as e:
            old_log_dir = log_dir
            log_dir = expfig.make_sequential_log_dir(
                None,
                subdirs=subdirs,
                use_existing_dir=log_config.log_dir.use_existing_dir,
                logger_file='output.log'
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

    def serialize_config(self, serialize=True):
        config_dir = self.log_dirs.get('config')

        if config_dir and serialize:
            self.config.serialize_to_dir(config_dir, use_existing_dir=True, with_default=True)

    @abstractmethod
    def _setup_algo(self, setup_algo):
        pass

    @abstractmethod
    def _compute_baselines(self, baseline='rbc'):
        return {}

    def train_and_evaluate(self):
        self.train()
        return self.evaluate()

    def train(self):
        self._set_trajectories(train=True)

        log_dir = self.log_dirs.get('train_log')
        train_output = self._train(log_dir)

        if log_dir:
            print(f'Logged results in dir: {os.path.abspath(log_dir)}')

        return train_output

    def _train(self, log_dir):
        pass

    def evaluate(self, log_dir=None, set_eval_trajectory=True):
        if set_eval_trajectory:
            self._set_trajectories(evaluate=True)

        output = self._evaluate()

        log_dir = log_dir or self.log_dirs.get('evaluate_log')

        if log_dir is not None:
            self.save_with_metadata(output, log_dir, 'log.csv')
            self.save_with_metadata(output.sum(), log_dir, 'log_total.csv')

            print(f'Logged results in dir:\n\t{os.path.abspath(log_dir)}')

        return output

    def _evaluate(self):
        return self.algo.run(verbose=self.config.verbosity > 0)

    @staticmethod
    def save_with_metadata(table, log_dir, fname):
        log_path = os.path.join(log_dir, fname)
        table.to_csv(log_path)

        metadata = Trainer.get_metadata(table)
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

        microgrid.reset()

        return microgrid

    @staticmethod
    def load_pretrained_policy(pretrained_policy_dir, self_config=None):
        pretrainer = RLTrainer.load(pretrained_policy_dir, setup_algo=False)

        if self_config is not None:
            symmetric_config_diff = self_config.symmetric_difference(pretrainer.config)
            if symmetric_config_diff:
                warnings.warn(f'Difference between config and pretrained config exists:\n'
                              f'{symmetric_config_diff.pprint(log_func=lambda x: None)}')

        return pretrainer.algo.learner

    @classmethod
    def load(cls, log_dir, additional_config=None, additional_garage_data=None, setup_algo=False):
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

        instance = cls(config=config, default=default, setup_algo=setup_algo, serialize_config=False)
        garage_data = instance.load_additional_data(log_dir, additional_garage_data)

        if garage_data:
            return instance, garage_data

        return instance

    def load_additional_data(self, log_dir, additional_garage_data):
        pass

    @classmethod
    def evaluate_last_epoch(cls, log_dir, suffix=lambda epoch: f'evaluate_log_epoch_{epoch}'):
        raise NotImplementedError(f'evaluate_last_epoch is implemented for subclasses of RLTrainer,'
                                  f'not {cls.__name__}')

    def _pre_env_setup(self):
        if self.config.env.net_load.use:
            self.env_class = NetLoadContinuousMicrogridEnv
            return {'slack_module': self.config.env.net_load.slack_module}

        return {}


class RLTrainer(Trainer):
    algo_name = 'rl'
    env_class: Union[ContinuousMicrogridEnv, DiscreteMicrogridEnv, NetLoadContinuousMicrogridEnv]
    rnd_model: Union[RNDModel, None]

    def _set_wandb_env_keys(self):
        return set_wandb_env_keys(
            username=self.config.context.wandb.username,
            api_key_file=self.config.context.wandb.api_key_file,
            search_paths=[Path(__file__).parent]
        )

    def wrap_env(self, env, **_):
        eval_env = deepcopy(env)
        env = self._setup_domain_randomization(env)
        env = self._setup_forced_genset(env)

        # We setup forced genset on eval env but not dr
        eval_env = self._setup_forced_genset(eval_env)

        wrapped_env = super().wrap_env(env, return_copy=False)
        wrapped_eval_env = super().wrap_env(eval_env, return_copy=False)

        return wrapped_env, wrapped_eval_env

    def _setup_domain_randomization(self, env):
        dr_config = self.config.env.domain_randomization

        if not dr_config.noise_std:
            return env

        return DomainRandomizationWrapperV2(env,
                                          noise_std=float(dr_config.noise_std),
                                          relative_noise=dr_config.relative_noise
                                          )

    def _setup_forced_genset(self, env):
        # This will fail if used in DQN algorithms

        forced_genset = self.config.env.forced_genset
        if not forced_genset or forced_genset.strip().lower() in ['none', 'null']:
            return env

        force_on = expfig.functions.str2bool(forced_genset)

        if isinstance(env, DomainRandomizationWrapper):
            self.logger.warning('ForcedGenset incompatible with DomainRandomization, DomainRandomization will be '
                                'removed.')

        return ForcedGensetWrapper(env, force_on)

    def _setup_algo(self, setup_algo=True):
        if not setup_algo:
            warnings.warn('Skipping algo setup.')
            self.sampler = None
            return None

        self.warn_custom_params()
        algo, self.sampler, self.rnd_model = self.setup_rl_algo()
        return algo

    def _compute_baselines(self, baseline='rbc'):
        if not baseline:
            return {}

        elif pd.api.types.is_list_like(baseline):
            return {bl: self._compute_baselines(bl) for bl in baseline}

        baseline_specific_config = expfig.functions.unflatten({
            'context.log_dir.parent': os.path.join(self.log_dirs.get('train_log', 'experiment_logs'), 'baselines'),
            'context.wandb.api_key_file': None,
            'microgrid.methods.set_forecaster.forecast_horizon': 23,
            'microgrid.methods.set_forecaster.forecaster': 'oracle'
        })

        baseline_config = self.config.copy().update(baseline_specific_config)

        if baseline.lower() == 'rbc':
            baseline_trainer = RBCTrainer(config=baseline_config)
        elif baseline.lower() == 'mpc':
            baseline_trainer = MPCTrainer(config=baseline_config)
        else:
            raise NameError(baseline)

        eval = baseline_trainer.evaluate()
        reward_cumsum = eval['balance', 0, 'reward'].cumsum()

        return reward_cumsum

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
            'is_tf_worker': False,
            'seed': get_seed()
        }

        if sampler_config.type == 'ray':
            return RaySampler(**sampler_kwargs)
        elif sampler_config.type == 'local':
            return LocalSampler(**sampler_kwargs)
        else:
            raise ValueError(f"Invalid sampler config type {sampler_config.type}, must be 'local' or 'ray'.")

    def _setup_rnd_model(self):
        rnd_config = self.config.algo.rnd
        if not rnd_config.intrinsic_reward_weight:
            return None
        return RNDModel(self.env.spec.observation_space.shape[0], **rnd_config)

    def warn_custom_params(self):
        algos = ['dqn', 'ddpg', 'ppo', 'pretrain']
        algos.remove(self.algo_name)
        for other_algo in algos:
            custom_in_other = self.config.algo[other_algo].symmetric_difference(
                self.config.default_config.algo[other_algo])

            if custom_in_other:
                log_func = lambda d: warnings.warn(f"Custom parameters for algorithm '{other_algo}' will be ignored "
                                                   f"with algo='{self.algo_name}': \n{d}")
                custom_in_other.pprint(indent=4, log_func=log_func)

    def _train(self, log_dir):
        if self.config.algo.package == 'garage':
            return self._train_garage(log_dir)
        else:
            raise ValueError(self.config.algo.package)

    def _train_garage(self, log_dir):
        log_config = self.config.context
        train_config = self.config.algo.train

        pymgrid_commit = subprocess.check_output(
            ["git", "describe", "--always"], cwd=Path(pymgrid.__file__).resolve().parent).strip().decode()

        wandb_run = self._setup_wandb(self.log_dirs.get('train_log', -1), pymgrid_commit=pymgrid_commit, pid=os.getpid())
        experiment_name = getattr(wandb_run, 'name', log_dir)

        @wrap_experiment(name=experiment_name,
                         snapshot_mode='gap_overwrite',
                         snapshot_gap=log_config.snapshot_gap,
                         archive_launch_repo=False,
                         log_dir=log_dir,
                         use_existing_dir=True,
                         wandb_run=wandb_run)
        def train(ctxt=None):
            garage_trainer = GarageTrainer(ctxt)

            self.update_trainer(garage_trainer)

            garage_trainer.setup(self.algo, self.env, self.callback)
            garage_trainer.train(n_epochs=train_config.n_epochs, batch_size=train_config.batch_size)

            self.env.close()

            print(f'Logged results in dir: {log_dir}')

        train()

    def callback(self, env, policy, epoch):
        if not self.config.context.disable_logging:
            log = self._evaluate(policy=policy)
            self._record_log(log)

    def _evaluate(self, env=None, policy=None):
        env = env or self.eval_env.unwrapped
        policy = policy or self.algo.policy

        obs = env.reset()
        done = False

        while not done:
            obs, reward, done, _ = env.step(policy.get_action(obs)[0])

        return env.get_log(drop_singleton_key=True)

    def _record_log(self, log):
        numbered_modules = log.columns.nlevels == 3

        with tabular.prefix('Test/'):
            tabular.record_many(log[('balance', 0) if numbered_modules else 'balance'].sum().to_dict())

            if self.has_wandb:
                log.columns = log.columns.map(lambda x: '_'.join(str(v) for v in x if str(v)))
                table = wandb.Table(dataframe=log.reset_index(names='Step'))
                tabular.record('EvalLog', table)

                titles = [
                    'Cumulative Reward',
                    'Cumulative Shaped Reward',
                ]

                reward_cumsum_y = [
                    f'balance{"_0"*numbered_modules}_reward',
                    f'balance{"_0" * numbered_modules}_shaped_reward'
                ]

                table_subset = log[reward_cumsum_y].cumsum()

                for baseline_name, baseline in self.baseline_rewards.items():
                    titles.append(f'{baseline_name.upper()} Relative Cumulative Reward')

                    table_subset[f'{baseline_name}_baseline'] = table_subset[reward_cumsum_y[0]] / baseline
                    table_subset[f'{baseline_name}_baseline'].iloc[:10] = float('nan')  # transient
                    tabular.record(f'{baseline_name.upper()}RelativeRewardSum', table_subset[f'{baseline_name}_baseline'].iloc[-1])

                table_subset = table_subset.reset_index(names='Step')

                for column, title in zip(table_subset.columns[1:], titles):
                    plot = wandb.plot.line(
                        table=wandb.Table(dataframe=table_subset[['Step', column]]),
                        x='Step',
                        y=column,
                        title=title
                    )

                    tabular.record(title.replace(' ', ''), plot)

    def update_trainer(self, trainer):
        pass

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

    @classmethod
    def evaluate_last_epoch(cls, log_dir, suffix=lambda epoch: f'evaluate_log_epoch_{epoch}'):
        trainer, experiment_stats = cls.load(log_dir, additional_garage_data='stats', setup_algo=False)
        last_epoch = experiment_stats.total_epoch

        expfig.logging.get_logger().info(f'Evaluating last saved epoch ({last_epoch}).')

        evaluate_log_dir = os.path.join(log_dir, suffix(last_epoch))
        trainer.check_logdir_existence(evaluate_log_dir)

        return trainer.evaluate(evaluate_log_dir)

    @staticmethod
    def check_logdir_existence(log_dir):

        logger = expfig.logging.get_logger()

        try:
            os.mkdir(log_dir)
        except FileExistsError:
            import time
            contents = [x.name for x in Path(log_dir).iterdir()]

            if contents:
                contents_str = '\n\t\t'.join(contents)
                contents_str = f'\nContains contents:\n\t\t{contents_str}'
            else:
                contents_str = '\nDirectory is empty.'

            logger.warning(f'Logging to directory that already exists:\n\t{log_dir}'
                           f'{contents_str}\nContinuing in five seconds.')
            time.sleep(5)
            logger.info('Continuing...')


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
        rnd_model = self._setup_rnd_model()
        return self._setup_rl_algo(qf, policy, exploration_policy, sampler), sampler, rnd_model

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
        rnd_model = self._setup_rnd_model()
        return self._setup_rl_algo(qf, policy, exploration_policy, sampler, rnd_model), sampler, rnd_model

    def _setup_policies(self):
        qf = self.setup_qf(self.env.spec, self.config.algo.policy.hidden_sizes)

        policy = self.setup_policy(self.env.spec,
                                   self.config.algo.policy.hidden_sizes,
                                   self.config.microgrid.methods.set_module_attrs.normalized_action_bounds,
                                   pretrained_policy=self.config.algo.policy.pretrained_policy,
                                   self_config=self.config)

        exploration_policy = AddOrnsteinUhlenbeckNoise(self.env.spec, policy,
                                                       sigma=self.config.algo.ddpg.policy.exploration.sigma,
                                                       theta=self.config.algo.ddpg.policy.exploration.theta)

        return qf, policy, exploration_policy

    @staticmethod
    def setup_qf(env_spec, hidden_sizes):
        return ContinuousMLPQFunction(env_spec=env_spec,
                                      hidden_sizes=hidden_sizes)

    @staticmethod
    def setup_policy(env_spec, hidden_sizes, action_bounds, pretrained_policy=None, self_config=None):
        if pretrained_policy is not None:
            return RLTrainer.load_pretrained_policy(pretrained_policy, self_config=self_config)

        if action_bounds == [-1, 1]:
            output_nonlinearity = torch.tanh
        elif action_bounds == [0, 1]:
            output_nonlinearity = torch.sigmoid
        else:
            warnings.warn(f"Unable to map microgrid action bounds '{action_bounds}' to output nonlinearity. "
                          f"Using linear activation.")

            output_nonlinearity = None

        return DeterministicMLPPolicy(env_spec, hidden_sizes=hidden_sizes, output_nonlinearity=output_nonlinearity)

    def _setup_rl_algo(self, qf, policy, exploration_policy, sampler, rnd_model):

        replay_buffer = PathBuffer(capacity_in_transitions=self.config.algo.replay_buffer.buffer_size)

        ddpg_params = dict(
            env_spec=self.env.spec,
            policy=policy,
            qf=qf,
            replay_buffer=replay_buffer,
            sampler=sampler,
            exploration_policy=exploration_policy,
            **self.config.algo.general_params,
            **self.config.algo.deterministic_params,
            **self.config.algo.ddpg.params
        )

        if rnd_model is None:
            return DDPG(**ddpg_params)

        return RNDDDPG(rnd_model=rnd_model, **ddpg_params)


class PPOTrainer(RLTrainer):
    algo_name = 'ppo'
    env_class = ContinuousMicrogridEnv

    def setup_rl_algo(self):
        policy = self.setup_policy(self.env.spec,
                                   self.config.algo.policy.hidden_sizes,
                                   self.config.microgrid.methods.set_module_attrs.normalized_action_bounds,
                                   tanhnormal=self.config.algo.ppo.tanhnormal,
                                   pretrained_policy=self.config.algo.policy.pretrained_policy,
                                   self_config=self.config)

        value_function = self.setup_vf(self.env.spec, self.config.algo.policy.hidden_sizes)
        sampler = self._setup_sampler(policy)
        rnd_model = self._setup_rnd_model()

        return self._setup_rl_algo(policy, value_function, sampler, rnd_model), sampler, rnd_model

    @staticmethod
    def setup_policy(env_spec, hidden_sizes, action_bounds, tanhnormal=False, pretrained_policy=None, self_config=None):
        if pretrained_policy is not None:
            return RLTrainer.load_pretrained_policy(pretrained_policy, self_config=self_config)

        if tanhnormal:
            from utils.kl_register import register_tanhnormal
            register_tanhnormal()

            if action_bounds != [-1, 1]:
                warnings.warn(f"Recommended action bounds for tanhnormal policy are [-1, 1] to correspond with "
                              f"range(tanh(x), have '{action_bounds}'")

            return TanhGaussianMLPPolicy(env_spec, hidden_sizes=hidden_sizes)

        return GaussianMLPPolicy(env_spec, hidden_sizes=hidden_sizes)

    @staticmethod
    def setup_vf(env_spec, hidden_sizes):
        return GaussianMLPValueFunction(env_spec, hidden_sizes)

    def _setup_rl_algo(self, policy, value_function, sampler, rnd_model):
        ppo_params = dict(
            env_spec=self.env.spec,
            policy=policy,
            value_function=value_function,
            sampler=sampler,
            **self.config.algo.general_params,
            **self.config.algo.ppo.params
        )

        if rnd_model is None:
            return PPO(**ppo_params)

        return RNDPPO(rnd_model=rnd_model, **ppo_params)


class PreTrainer(RLTrainer):
    algo_name = 'pretrain'
    env_class = ContinuousMicrogridEnv
    expert = None

    def setup_rl_algo(self):
        learner, qf_or_vf = self._setup_learner()
        self.expert = self._get_expert()
        sampler = self._setup_sampler(self.expert)

        return self._setup_rl_algo(learner, self.expert, sampler, qf_or_vf), sampler, None

    def _setup_learner(self):
        algo_to_pretrain = self.config.algo.pretrain.algo_to_pretrain

        env_spec = self.env.spec
        hidden_sizes = self.config.algo.policy.hidden_sizes
        action_bounds = self.config.microgrid.methods.set_module_attrs.normalized_action_bounds

        if algo_to_pretrain == 'ddpg':
            policy = DDPGTrainer.setup_policy(env_spec, hidden_sizes, action_bounds)
            qf_or_vf = {'qf': DDPGTrainer.setup_qf(env_spec, hidden_sizes)}
        elif algo_to_pretrain == 'ppo':
            policy = PPOTrainer.setup_policy(env_spec, hidden_sizes, action_bounds)
            qf_or_vf = {'value_function': PPOTrainer.setup_vf(env_spec, hidden_sizes)}
        else:
            raise ValueError(f"config.pretrain.algo_to_pretrain must be 'ddpg' or 'ppo', not '{algo_to_pretrain}'.")

        return policy, qf_or_vf

    def _get_expert(self):
        from pretrain import Expert
        return Expert(expert_type=self.config.algo.pretrain.pretrain_algo,
                      episodes_per_batch=self.config.algo.pretrain.params.episodes_per_batch,
                      additional_config=self.config.algo.pretrain.additional_config)

    def _setup_rl_algo(self, learner, expert, sampler, qf_or_vf):
        from pretrain import BC

        return BC(
            env_spec=self.env.spec,
            learner=learner,
            batch_size=self.config.algo.train.batch_size,
            source=expert,
            sampler=sampler,
            loss=self.config.algo.pretrain.params.loss,
            policy_lr=self.config.algo.pretrain.params.policy_lr,
            **qf_or_vf,  # qf OR value_function
            **self.config.algo.general_params,  # discount
            **self.config.algo.ddpg.params,  # target_update_tau
        )

    def update_trainer(self, trainer):
        self.expert.set_stats(trainer._stats)

    def _evaluate(self, env=None, **_):
        return super()._evaluate(env=env, policy=self.algo.learner)


class MPCTrainer(Trainer):
    algo_name = 'mpc'

    def _setup_algo(self, setup_algo=True):
        if not setup_algo:
            warnings.warn('Skipping algo setup.')
            return None

        return ModelPredictiveControl(self.microgrid)

    def _train(self, log_dir):
        return self.evaluate(log_dir, set_eval_trajectory=False)


class RBCTrainer(Trainer):
    algo_name = 'rbc'

    def _setup_algo(self, setup_algo=True):
        if not setup_algo:
            warnings.warn('Skipping algo setup.')
            return None

        return RuleBasedControl(self.microgrid)

    def _train(self, log_dir):
        return self.evaluate(log_dir, set_eval_trajectory=False)


if __name__ == '__main__':
    Trainer().train_and_evaluate()
