from garage.torch.algos.dqn import DQN
from garage.envs import GymEnv
from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler, RaySampler
from garage.trainer import Trainer
from garage.np.exploration_policies import EpsilonGreedyPolicy
from garage.torch.policies import DiscreteQFArgmaxPolicy
from garage.torch.q_functions import DiscreteMLPQFunction
from garage.replay_buffer import PathBuffer

from pymgrid.microgrid.envs import DiscreteMicrogridEnv

import sys


# PARAMETERS TO CHANGE:
sampler_type = 'local' # 'local' or 'ray'
name = "first_experiment"
log_dir = f'{name}'

hyperparams = dict(n_epochs=800,
                   steps_per_epoch=32,
                   sampler_batch_size=10000,
                   lr=1e-4,
                   discount=0.99,
                   min_buffer_size=int(1e4),
                   n_train_steps=500,
                   target_update_freq=2,
                   buffer_batch_size=32,
                   max_epsilon=1.0,
                   min_epsilon=0.25,
                   decay_ratio=0.5,
                   buffer_size=int(2*1e5),
                   hiddens=(512, ),
                   hidden_channels=(32, 64, 64),
                   kernel_sizes=(8, 4, 3),
                   strides=(4, 2, 1),
                   clip_gradient=10)

if name is None:
    raise ValueError('Must define a name for this experiment above. '
                     'log_dir may also be modified if a different directory is desired.')

@wrap_experiment(name=name,
                 snapshot_mode='gap',
                 snapshot_gap=50,
                 log_dir=log_dir)
def discrete_dqn(ctxt=None, seed=1):
    set_seed(seed)
    trainer = Trainer(ctxt)

    env = DiscreteMicrogridEnv.from_scenario(microgrid_number=2)
    env = GymEnv(env, max_episode_length=len(env))

    qf = DiscreteMLPQFunction(env_spec=env.spec, hidden_sizes=[8, 5])
    policy = DiscreteQFArgmaxPolicy(env_spec=env.spec, qf=qf)

    n_epochs = hyperparams['n_epochs']
    steps_per_epoch = hyperparams['steps_per_epoch']
    sampler_batch_size = hyperparams['sampler_batch_size']
    num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size
    replay_buffer = PathBuffer(
        capacity_in_transitions=hyperparams['buffer_size'])

    exploration_policy = EpsilonGreedyPolicy(
        env_spec=env.spec,
        policy=policy,
        total_timesteps=num_timesteps,
        max_epsilon=hyperparams['max_epsilon'],
        min_epsilon=hyperparams['min_epsilon'],
        decay_ratio=hyperparams['decay_ratio'])


    if sampler_type == 'local':
        sampler = LocalSampler(
            agents=exploration_policy,
            envs=env,
            max_episode_length=env.spec.max_episode_length,
            n_workers=16,
            is_tf_worker=False)

    elif sampler_type == 'ray':
        sampler = RaySampler(agents=exploration_policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=False)
    else:
        raise ValueError(sampler_type)

    algo = DQN(env_spec=env.spec,
               policy=policy,
               qf=qf,
               exploration_policy=exploration_policy,
               replay_buffer=replay_buffer,
               sampler=sampler,
               steps_per_epoch=steps_per_epoch,
               qf_lr=hyperparams['lr'],
               clip_gradient=hyperparams['clip_gradient'],
               discount=hyperparams['discount'],
               min_buffer_size=hyperparams['min_buffer_size'],
               n_train_steps=hyperparams['n_train_steps'],
               target_update_freq=hyperparams['target_update_freq'],
               buffer_batch_size=hyperparams['buffer_batch_size'],
               double_q=False)

    trainer.setup(algo, env)
    trainer.train(n_epochs=n_epochs, batch_size=sampler_batch_size)
    env.close()
    print('Training completed')
    print(f'Saved results in dir: {trainer._snapshotter.snapshot_dir}')

discrete_dqn()
