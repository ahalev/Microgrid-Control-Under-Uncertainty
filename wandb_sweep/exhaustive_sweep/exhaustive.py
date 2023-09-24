import os

from pymgrid.microgrid import trajectory
import wandb
import yaml

from expfig.functions import unflatten, flatten
from pathlib import Path

from wandb_sweep import Sweep

DEFAULT_CONFIG = Path(__file__).parent / 'exhaustive_sweep_config.yaml'

CONFIG_REPLACEMENTS = {
    'FixedLengthStochasticTrajectory(trajectory_length=720)': trajectory.FixedLengthStochasticTrajectory(trajectory_length=720)
}


class ExhaustiveSweep(Sweep):
    cls_dir = Path(__file__).parent

    def __init__(self, config=None, default=DEFAULT_CONFIG):
        super().__init__(config=config, default=default)

        self.base_config = self.get_base_config()
        self.add_scenario()

    def get_base_config(self):
        if self.sweep_id is not None and self.get_base_config_file().exists():
            # # running an agent, pass
            return

        api = wandb.Api()
        runs = api.runs(path='ucd-cnml/gridrl')

        for run in runs:
            if self.meta_config.config_from_run in (run.name, run.id):
                self.logger.info(f'Found run for base config:\n\t{run.name=}\n\t{run.id=}')
                return cleanup_config(run.config, {'context.wandb.api_key_file': self.meta_config.wandb_setup.api_key_file})

        raise NameError(f'No run found to match {self.meta_config.config_from_run=}')

    def add_scenario(self):
        if self.base_config is not None:
            self.add_parameter('microgrid.config.scenario', self.base_config['microgrid']['config']['scenario'])

    def setup(self):
        launch_agent = self.meta_config.launch_agent
        self.meta_config.launch_agent = False

        super().setup()
        self._setup_base_config()

        self.meta_config.launch_agent = launch_agent

        if launch_agent:
            self.launch_agent_v2()

    def _setup_base_config(self):
        base_config_file = self.get_base_config_file()

        assert base_config_file.parent.parent.exists()
        base_config_file.parent.mkdir(exist_ok=True)

        with base_config_file.open('w') as f:
            yaml.safe_dump(self.base_config, f)

        self.logger.info(f'Dumped based on config to {base_config_file}')

        return base_config_file

    def get_base_config_file(self):
        assert self.sweep_id is not None
        stripped_sweep_id = self.sweep_id.rpartition('/')[-1]

        log_dir_parent = os.environ.get('WANDB_DIR', os.getcwd())
        base_config_file = os.path.join(
            log_dir_parent,
            'wandb',
            f'sweep-{stripped_sweep_id}',
            'based_on_config.yaml'
        )

        return Path(base_config_file).resolve()

    def set_config_env_var(self):
        base_config_file = self.get_base_config_file()
        assert base_config_file.exists(), f'Base config file {base_config_file} not found'

        os.environ.setdefault('BASED_ON_CONFIG', str(base_config_file))

    def launch_agent_v2(self, sweep_id=None, count=5):
        if sweep_id is not None:
            self.sweep_id = sweep_id

        self.set_config_env_var()
        # TODO if this doesn't work, you need to pass `env` to subprocess.Popen
        super().launch_agent_v2(count=count)


def cleanup_config(config, additional_flattened=None):
    if additional_flattened is None:
        additional_flattened = {}

    flattened = flatten(config)
    replaced = {key: CONFIG_REPLACEMENTS.get(str(value), value) for key, value in flattened.items()}
    replaced.update(additional_flattened)

    return unflatten(replaced)


if __name__ == '__main__':
    ExhaustiveSweep().setup()
