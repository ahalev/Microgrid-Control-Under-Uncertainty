import os
import pandas as pd
import wandb

from expfig import Config
from expfig.functions import flatten
from pathlib import Path
from dowel import set_wandb_env_keys


os.system("ulimit -n 4096")

DEFAULT_CONFIG = Path(__file__).parent / 'ppo_sweep_config.yaml'


class Sweep:
    def __init__(self, config=None):
        self.config = Config(config, default=DEFAULT_CONFIG)

        self.meta_config = self.config.pop('meta')

        if not set_wandb_env_keys(**self.meta_config.wandb_setup, search_paths=['.']):
            raise RuntimeError('wandb not initialized!')

        self.sweep_id = self.meta_config.sweep_id

    def add_parameter(self, parameter_name, value):
        parameter_name = tuple(parameter_name.split('.'))

        if pd.api.types.is_dict_like(value):
            _value = {**value}
        elif pd.api.types.is_list_like(value):
            _value = {'values': value}
        else:
            _value = {'value': value}

        self.config.parameters[parameter_name] = _value

    def setup(self):
        config = self.config.to_dict()
        config['parameters'] = flatten(config['parameters'], levels=-1)
        self.sweep_id = wandb.sweep(config)

        if self.meta_config.launch_agent:
            self.launch_agent()

    def launch_agent(self, sweep_id=None, count=5):
        sweep_id = sweep_id or self.sweep_id
        if sweep_id is None:
            raise ValueError('sweep_id not found')

        wandb.agent(sweep_id, count=None)


if __name__ == '__main__':
    sweep = Sweep({"meta.launch_agent": True})
    sweep.setup()
