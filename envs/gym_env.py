from garage.envs import GymEnv as _GarageGymEnv


class GymEnv(_GarageGymEnv):
    def __init__(self, env, is_image=False, max_episode_length=None):
        super().__init__(env, is_image=is_image, max_episode_length=max_episode_length)
        self._cast_dtypes()

    def _cast_dtypes(self):
        self._observation_space.dtype = self._env.observation_space.dtype
        self._action_space.dtype = self._env.action_space.dtype
