from gym import Env
from typing import Union

from pymgrid.envs import DiscreteMicrogridEnv, ContinuousMicrogridEnv
from pymgrid.forecast import GaussianNoiseForecaster


class DomainRandomizationWrapper(Env):
    def __init__(self,
                 env: Union[DiscreteMicrogridEnv, ContinuousMicrogridEnv],
                 noise_std,
                 relative_noise=True):
        self._env = env
        self._noisemakers, self._og_time_series = self._get_noisemakers(noise_std,
                                                                       False,
                                                                       relative_noise=relative_noise)

    def _get_noisemakers(self, noise_std, increase_uncertainty, relative_noise):
        noisemakers = []
        ts = []
        for module in self._env.modules.iterlist():
            try:
                time_series = module.time_series
                ts.append(time_series)
            except AttributeError:
                noisemakers.append(None)
                ts.append(None)
            else:
                noisemakers.append(
                    self._get_time_series_module_noisemaker(module, noise_std, increase_uncertainty, relative_noise)
                )

        return noisemakers, ts

    def _get_time_series_module_noisemaker(self, module, noise_std, increase_uncertainty, relative_noise):
        return GaussianNoiseForecaster(
                noise_std=noise_std,
                observation_space=module.observation_space,
                forecast_shape=module.time_series.shape,
                time_series=module.time_series,
                increase_uncertainty=increase_uncertainty,
                relative_noise=relative_noise
            )

    def step(self, action):
        return self._env.step(action)

    def reset(self, **kwargs):
        self.randomize_timeseries()
        return self._env.reset()

    def randomize_timeseries(self):
        for module, noisemaker, og_ts in zip(self._env.modules.iterlist(), self._noisemakers, self._og_time_series):
            if noisemaker is None:
                assert not hasattr(module, 'time_series')
                continue

            assert hasattr(module, 'time_series')

            module.time_series = noisemaker(None, og_ts, len(og_ts))

    def render(self):
        return self._env.render()

    def __getattr__(self, item):
        return getattr(self._env, item)

    def __setattr__(self, key, value):
        if key != '_env' and hasattr(self._env, key):
            setattr(self._env, key, value)
        else:
            super().__setattr__(key, value)

    def __getstate__(self):
        raise RuntimeError

    @classmethod
    def to_yaml(cls, dumper, data):
        raise RuntimeError

    @classmethod
    def from_yaml(cls, loader, node):
        raise RuntimeError

    @property
    def unwrapped(self):
        return self._env
