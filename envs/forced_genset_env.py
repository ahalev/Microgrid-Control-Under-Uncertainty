import functools

from envs import GymEnv, parse_potential_gym_env
from envs.gym_env import AttrPassthroughMixin
from gym.core import Wrapper


class ForcedGensetWrapper(GymEnv):
    def __init__(self,
                 env,
                 on=False,
                 is_image=False,
                 max_episode_length=None,
                 **kwargs
                 ):

        self._post_init = False
        super().__init__(**parse_potential_gym_env(env, is_image, max_episode_length))

        self.on = on
        self.unwrapped.convert_action = convert_action_decorator(on)(self.unwrapped.convert_action)

        self._post_init = True

    def _setup_setup_ranges(self):
        return None, None

    # def get_new_convert_action(self):
    #     old_convert_action = self.unwrapped.__class__.convert_action
    #
    #     def new_convert_action(obj, action, to_microgrid=True, normalize=False):
    #         converted = old_convert_action(obj, action, to_microgrid=to_microgrid, normalize=normalize)
    #         if not to_microgrid:
    #             return converted
    #         else:
    #             raise RuntimeError
    #
    #     return new_convert_action

    def reset(self, **kwargs):
        return super().reset()

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
        return super().__getstate__()

    def __setstate__(self, state):

        self.__init__(
            env=state['_env'],
            max_episode_length=state['_max_episode_length']
        )


def convert_action_decorator(on=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(action, to_microgrid=True, normalize=False):
            converted = func(action, to_microgrid=to_microgrid, normalize=normalize)

            if to_microgrid and not normalize and 'genset' in converted:
                for act in converted['genset']:
                    act[0] = on

            return converted

        return wrapper
    return decorator


class StepCounter:
    def __init__(self, step=0):
        self.step = step

    def update(self, n=1):
        self.step += n

    def __call__(self):
        return self.step
