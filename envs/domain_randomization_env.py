from envs import GymEnv, parse_potential_gym_env

from pymgrid import dry_run
from pymgrid.forecast import GaussianNoiseForecaster


class DomainRandomizationWrapper(GymEnv):
    def __init__(self,
                 env,
                 noise_std,
                 relative_noise=True,
                 is_image=False,
                 max_episode_length=None):

        self._post_init = False
        super().__init__(**parse_potential_gym_env(env, is_image, max_episode_length))

        self._noise_std = noise_std
        self._relative_noise = relative_noise

        self._noisemakers, self._og_time_series = self._get_noisemakers(False)
        self._post_init = True

    def _get_noisemakers(self, increase_uncertainty):
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
                    self._get_time_series_module_noisemaker(module, increase_uncertainty)
                )

        return noisemakers, ts

    def _get_time_series_module_noisemaker(self, module, increase_uncertainty):
        observation_space = self._extend_obs_space(module)

        return GaussianNoiseForecaster(
                noise_std=self._noise_std,
                observation_space=observation_space,
                forecast_shape=module.time_series.shape,
                time_series=module.time_series,
                increase_uncertainty=increase_uncertainty,
                relative_noise=self._relative_noise
            )

    def _extend_obs_space(self, module):
        with dry_run(module) as m:
            m.forecast_horizon = len(module.time_series) - 1
            return m.observation_space

    def reset(self, **kwargs):
        self.randomize_timeseries()
        return super().reset()

    def randomize_timeseries(self):
        for module, noisemaker, og_ts in zip(self._env.modules.iterlist(), self._noisemakers, self._og_time_series):
            if noisemaker is None:
                assert not hasattr(module, 'time_series')
                continue

            assert hasattr(module, 'time_series')

            module.time_series = noisemaker(None, og_ts, len(og_ts))

    def reset_timeseries(self):
        for module, og_ts in zip(self._env.modules.iterlist(), self._og_time_series):
            if og_ts is not None:
                module.time_series = og_ts

    @classmethod
    def to_yaml(cls, dumper, data):
        raise RuntimeError

    @classmethod
    def from_yaml(cls, loader, node):
        raise RuntimeError

    @property
    def noise_std(self):
        return self._noise_std

    @noise_std.setter
    def noise_std(self, value):
        self._noise_std = value
        self._noisemakers, self._og_time_series = self._get_noisemakers(False)

    @property
    def relative_noise(self):
        return self._relative_noise

    @relative_noise.setter
    def relative_noise(self, value):
        self._relative_noise = value
        self._noisemakers, self._og_time_series = self._get_noisemakers(False)

    @property
    def max_episode_length(self):
        return self.spec.max_episode_length

    def __getstate__(self):
        self.reset_timeseries()
        return super().__getstate__()

    def __setstate__(self, state):

        self.__init__(
            env=state['_env'],
            noise_std=state['_noise_std'],
            relative_noise=state['_relative_noise'],
            max_episode_length=state['_max_episode_length']
        )
