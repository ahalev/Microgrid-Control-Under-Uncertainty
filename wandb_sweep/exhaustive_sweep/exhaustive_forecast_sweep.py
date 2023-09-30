from pathlib import Path

from wandb_sweep.exhaustive_sweep import ExhaustiveSweep


if __name__ == '__main__':
    forecast_default = Path(__file__).parent / 'forecast_sweep_config.yaml'
    ExhaustiveSweep(default=forecast_default).setup()
