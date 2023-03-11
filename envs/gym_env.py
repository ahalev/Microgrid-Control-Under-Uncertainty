from garage.envs import GymEnv as _GarageGymEnv


class GymEnv(_GarageGymEnv):
    def __init__(self, env, is_image=False, max_episode_length=None):
        super().__init__(env, is_image=is_image, max_episode_length=max_episode_length)
        self._cast_dtypes()

    def _cast_dtypes(self):
        self._observation_space.dtype = self._env.observation_space.dtype
        self._action_space.dtype = self._env.action_space.dtype

    def __setattr__(self, key, value):
        # This is a hack, but this is the last thing initialized in super().__init__.
        # Prevents conflicts in GymEnv attributes (albeit only those in defined in __init__) and anything in 'env'.
        post_init = hasattr(self, '_env_info')

        try:
            self.__getattribute__(key)
            toplevel_attr = True
        except AttributeError:
            toplevel_attr = False

        if post_init and not toplevel_attr and hasattr(self._env, key):
            setattr(self._env, key, value)
        else:
            super().__setattr__(key, value)
