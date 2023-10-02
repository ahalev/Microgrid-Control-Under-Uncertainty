import os
import subprocess

import pandas as pd
import wandb

from expfig import Config, get_logger
from expfig.functions import flatten
from dowel import set_wandb_env_keys

from pathlib import Path
from sklearn.model_selection import ParameterGrid


from wandb_sweep.agent_subprocess import run_and_terminate_process, kill_hanging

os.system("ulimit -n 4096")

DEFAULT_CONFIG = Path(__file__).parent / 'ppo_sweep_config.yaml'


class Sweep:
    cls_dir = Path(__file__).parent
    agent = 'agent.py'

    def __init__(self, config=None, default=DEFAULT_CONFIG):
        self.config = Config(config, default=default)

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

    def check_params(self, params):
        too_long = {k: v for k, v in params.items() if len(v) > 1}
        if not too_long:
            return params

        # One was in default:
        for k, v in too_long.items():
            split_key = tuple(k.split('.'))
            default = self.config.default_config.parameters[split_key]

            if 'value' in default:
                v.pop('value')
            elif 'values' in default:
                v.pop('values')
            else:
                raise RuntimeError

            assert len(v) == 1
            params[k] = v

        return params

    def setup(self):
        config = self.config.to_dict()
        config['parameters'] = self.check_params(flatten(config['parameters'], levels=-1))
        self.sweep_id = wandb.sweep(config)

        api_settings = wandb.Api().settings
        sweep_id = os.path.join(api_settings['entity'], api_settings['project'], self.sweep_id)

        launch_cmd = 'Launch agents with:\n' \
                     f'cd {self.cls_dir.resolve()} && python {self.agent} --meta.sweep_id {sweep_id}'

        self.logger.info(launch_cmd)

        if self.meta_config.launch_agent:
            self.launch_agent_v2()

    def commands_list(self, ignore_missing_env_vars=True):
        """
        Returns a list of commands that will be run in this sweep. Order will likely different from order
        runs are dispatched via `wandb agent`.
        """
        sweep = wandb.Api().sweep(f'ucd-cnml/gridrl/{self.sweep_id}')
        command = sweep.config['command']

        # one of these will exist, otherwise sweep construction would have failed
        flat_params = {k: v.get('values', [v.get('value')]) for k, v in sweep.config['parameters'].items()}
        parameter_grid = ParameterGrid(flat_params)

        commands = []
        for params in parameter_grid:
            commands.append(fill_command(command, params, sweep.config['program'],
                                         ignore_missing_env_vars=ignore_missing_env_vars))

        return commands

    def launch_agent(self, sweep_id=None, count=5):
        sweep_id = sweep_id or self.sweep_id
        if sweep_id is None:
            raise ValueError('sweep_id not found')

        wandb.agent(sweep_id, count=count)

    def launch_agent_v2(self, sweep_id=None):
        sweep_id = sweep_id or self.sweep_id
        if sweep_id is None:
            raise ValueError('sweep_id not found')

        command = f'wandb agent {sweep_id} --count 1'.split()

        for j in range(self.meta_config.agent_count):
            with run_and_terminate_process(command, stdout=subprocess.PIPE, text=True) as proc:
                self.logger.info(f'Running process {j} of {self.meta_config.agent_count}:\t{" ".join(proc.args)}')
                kill_hanging(proc, timeout=self.meta_config.agent_timeout)


def fill_command(command, params, program, ignore_missing_env_vars=True):
    new_command = []
    args_str = construct_args(params)

    for component in command:
        if '$' not in component:
            new_command.append(component)
        elif '${program}' in component:
            new_command.append(component.replace('${program}', program))
        elif '${envvar:' in component:
            new_command.append(replace_env_vars(component, ignore_missing_env_vars))
        elif '${args}' in component:
            new_command.append(args_str)
        else:
            raise ValueError(f"Unrecognized wildcard in {component}.")

    return ' '.join(new_command)


def construct_args(args_dict):
    args_list = [f'"--{k}={v}"' for k, v in args_dict.items()]
    return ' '.join(args_list)


def replace_env_vars(env_var_string, ignore_missing=True):
    bracket_splits = [x.split('}') for x in env_var_string.split('${')][1:]

    new_splits = []

    for j, split in enumerate(bracket_splits):
        inner_split = []
        for val in split:
            try:
                inner_split.append(get_env_var(val))
            except KeyError:
                if not ignore_missing:
                    raise
                else:
                    inner_split.append(f'${{{val}}}')

        new_splits.append(''.join(inner_split))

    return ''.join(new_splits)



def get_env_var(single_env_var_str):
    envvar_name = single_env_var_str.partition('envvar:')[-1]
    if not envvar_name:
        return single_env_var_str

    return os.environ[envvar_name]


if __name__ == '__main__':
    Sweep().setup()
