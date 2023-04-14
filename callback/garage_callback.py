from dowel import tabular, Histogram


class GarageCallback:
    def __init__(self, num_eval_episodes, max_episode_length):
        self.num_eval_episodes = num_eval_episodes
        self.max_episode_length = max_episode_length

        self._num_resets = 0
        self._steps_in_episode = 0
        self.actions = []

    def reset(self):
        self._num_resets += 1
        self._steps_in_episode = 0

    def step(self, action, done, **kwargs):
        self.actions.append(action)
        self._steps_in_episode += 1
        if self._num_resets == self.num_eval_episodes and (self._steps_in_episode == self.max_episode_length or done):
            self._record()
            self._num_resets = 0

    def _record(self):
        actions_hist = Histogram(self.actions)
        tabular.record('Callback/ActionDistribution', actions_hist)
