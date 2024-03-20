import functools

from envs.gym_env import AttrPassthroughMixin
from gym.core import Wrapper


class ForcedGensetWrapper(Wrapper, AttrPassthroughMixin):
    def __init__(self,
                 env,
                 on=False,
                 *args,
                 **kwargs
                 ):

        self._post_init = False
        super().__init__(env, *args, **kwargs)

        self.on = on

        self.env.convert_action = convert_action_decorator(on)(self.env.convert_action)
        self.step_counter = 0
        self._post_init = True

    def unwrapped(self):
        self.env.convert_action = self.env.convert_action.__wrapped__
        return super().wrapped

    def step(self, action):
        self.step_counter += 1
        if self.step_counter % 500 == 0:
            print(f'{self.step_counter=}')

        return self.env.step(action)

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
