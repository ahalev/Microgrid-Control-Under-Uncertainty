from pathlib import Path

from wandb_sweep.exhaustive_sweep import ExhaustiveSweep


if __name__ == '__main__':
    battery_default = Path(__file__).parent / 'battery_sweep_config.yaml'
    ExhaustiveSweep(default=battery_default).setup()
