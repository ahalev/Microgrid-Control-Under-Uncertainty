from envs import GymEnv

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
        super().__init__(**self.parse_env(env, is_image, max_episode_length))

        self._noise_std = noise_std
        self._relative_noise = relative_noise

        self._noisemakers, self._og_time_series = self._get_noisemakers(False)
        self._post_init = True

    def parse_env(self, env, is_image, max_episode_length):
        if isinstance(env, GymEnv):
            is_image = env.observation_space.__class__.__name__ == 'Image'
            max_episode_length = env.spec.max_episode_length
            env = env.unwrapped

        return {'env': env, 'is_image': is_image, 'max_episode_length': max_episode_length}

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
        try:
            self.__getattribute__(key)
            toplevel_attr = True
        except AttributeError:
            toplevel_attr = False

        if key != '_post_init' and self._post_init and not toplevel_attr and hasattr(self._env, key):
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
