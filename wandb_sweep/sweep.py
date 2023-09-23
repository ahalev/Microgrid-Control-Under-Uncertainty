import os
import subprocess

import pandas as pd
import wandb

from expfig import Config, get_logger
from expfig.functions import flatten
from pathlib import Path
from dowel import set_wandb_env_keys

from sweep import run_and_terminate_process, kill_hanging

os.system("ulimit -n 4096")

DEFAULT_CONFIG = Path(__file__).parent / 'ppo_sweep_config.yaml'


class Sweep:
    def __init__(self, config=None):
        self.config = Config(config, default=DEFAULT_CONFIG)

        self.meta_config = self.config.pop('meta')

        if not set_wandb_env_keys(**self.meta_config.wandb_setup, search_paths=['.']):
            raise RuntimeError('wandb not initialized!')

        self.sweep_id = self.meta_config.sweep_id

        self.add_parameter('microgrid.config.scenario', self.meta_config.scenario)

        self.logger = get_logger()

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

        api_settings = wandb.Api().settings
        sweep_id = os.path.join(api_settings['entity'], api_settings['project'], self.sweep_id)

        launch_cmd = 'Launch agents with:\n' \
                     f'cd {Path(__file__).parent.resolve()} && python agent.py --meta.sweep_id {sweep_id}'

        self.logger.info(launch_cmd)

        if self.meta_config.launch_agent:
            self.launch_agent()

    def launch_agent(self, sweep_id=None, count=5):
        sweep_id = sweep_id or self.sweep_id
        if sweep_id is None:
            raise ValueError('sweep_id not found')

        wandb.agent(sweep_id, count=count)

    def launch_agent_v2(self, sweep_id=None, count=5):
        sweep_id = sweep_id or self.sweep_id
        if sweep_id is None:
            raise ValueError('sweep_id not found')

        command = f'wandb agent {sweep_id} --count 1'.split()

        for j in range(count):
            with run_and_terminate_process(command, stdout=subprocess.PIPE, text=True) as proc:
                self.logger.info(f'Running process {j} of {count}:\t{proc}')
                kill_hanging(proc, timeout=self.meta_config.agent_timeout)


if __name__ == '__main__':
    Sweep().setup()
