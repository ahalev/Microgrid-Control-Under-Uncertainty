# Roughly following the API here:
# https://docs.ray.io/en/latest/rllib/rllib-training.html#basic-python-api

import ray
from ray.rllib.algorithms import apex_dqn
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from pymgrid.envs import DiscreteMicrogridEnv

hyperparams = dict(
    training_steps=10,
    num_workers=4,
    rollout_fragment_length=8760,
    hiddens=[512],
    initial_epsilon=1.0,
    final_epsilon=0.1,
    epsilon_timesteps=10000
)

env_config = {
    "env": "DiscreteMicrogridEnv",
    "env_config": {"microgrid_number": 2},
}

training_config = {
    "train_batch_size": hyperparams["num_workers"]*hyperparams["rollout_fragment_length"],
    "hiddens": hyperparams["hiddens"]
}

exploration_config = {
    "initial_epsilon": hyperparams["initial_epsilon"],
    "final_epsilon": hyperparams["final_epsilon"],
    "epsilon_timesteps": hyperparams["epsilon_timesteps"],
    "num_workers": hyperparams["num_workers"]
                      }

rollout_config = {
    "rollout_fragment_length": hyperparams["rollout_fragment_length"]
}


def microgrid_env_creator(env_config):
    if "microgrid_number" not in env_config:
        raise ValueError("env_config must have a microgrid number.")
    return DiscreteMicrogridEnv.from_scenario(microgrid_number=env_config["microgrid_number"])


def set_config(env_config, training_config, exploration_config, rollout_config):
    dqn_config = apex_dqn.ApexDQNConfig()
    dqn_config.training(**training_config)
    dqn_config.exploration(exploration_config=exploration_config)
    dqn_config.environment(**env_config)
    dqn_config.rollouts(**rollout_config)
    return dqn_config


def train(algo, training_steps):
    for j in range(training_steps):
        result = algo.train()
        print(pretty_print(result))

    return algo.evaluate()


ray.init()
register_env("DiscreteMicrogridEnv", microgrid_env_creator)
config = set_config(env_config, training_config, exploration_config, rollout_config)
algo = apex_dqn.ApexDQN(config=config, env="DiscreteMicrogridEnv")
eval = train(algo, hyperparams["training_steps"])

pretty_print(eval)

