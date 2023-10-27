import gym
from garage.envs import GymEnv as _GarageGymEnv


class AttrPassthroughMixin:
    _post_init: bool
    env: 'gym.Env'

    def __setattr__(self, key, value):
        try:
            self.__getattribute__(key)
            toplevel_attr = True
        except AttributeError:
            toplevel_attr = False

        if key != '_post_init' and self._post_init and not toplevel_attr and hasattr(self.env, key):
            setattr(self.env, key, value)
        else:
            super().__setattr__(key, value)

    def __len__(self):
        return len(self.env)


class GymEnv(_GarageGymEnv, AttrPassthroughMixin):
    def __init__(self, env, is_image=False, max_episode_length=None):
        self._post_init = False
        super().__init__(env, is_image=is_image, max_episode_length=max_episode_length)
        self._post_init = True
        self._cast_dtypes()

    def _cast_dtypes(self):
        self._observation_space.dtype = self._env.observation_space.dtype
        self._action_space.dtype = self._env.action_space.dtype

    def unwrap(self):
        return self._env

    @property
    def unwrapped(self):
        try:
            return self._env.unwrap()
        except AttributeError:
            return self._env

    @property
    def env(self):
        return self._env


def parse_potential_gym_env(env, is_image, max_episode_length):
    if isinstance(env, GymEnv):
        is_image = env.observation_space.__class__.__name__ == 'Image'
        max_episode_length = env.spec.max_episode_length
        env = env.unwrapped

    return {'env': env, 'is_image': is_image, 'max_episode_length': max_episode_length}
