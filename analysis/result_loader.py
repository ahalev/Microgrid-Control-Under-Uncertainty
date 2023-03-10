import numpy as np
import pandas as pd
import pymgrid
import seaborn as sns
import json
import yaml
import warnings


from expfig import Namespacify
from matplotlib import pyplot as plt
from pathlib import Path


pymgrid.add_pymgrid_yaml_representers()


class ResultLoader(Namespacify):
    def __init__(self, result_dir, save_dir=None):
        super().__init__(self._load_results(Path(result_dir)))

        self.passed_result_dir = Path(result_dir)
        self.save_dir = Path(save_dir) if save_dir else None
        self.evaluate_logs = self.locate_deep_key('evaluate_log')

    def _load_results(self, directory):
        results = {}
        for contents in directory.iterdir():
            if contents.is_dir():
                results[contents.stem] = self._load_results(contents)
            elif contents.suffix == '.yaml':
                results[contents.stem] = yaml.safe_load(contents.open('r'))
            elif contents.suffix == '.csv':
                results[contents.stem] = self._read_pandas(contents, pd.read_csv)
            elif contents.suffix == '.xlsx':
                results[contents.stem] = self._read_pandas(contents, pd.read_excel)
            elif contents.suffix == '.tag':
                continue
            else:
                warnings.warn(f'Ignoring file of unrecognized type {contents.name}')

        return results

    def _read_pandas(self, contents, load_func):
        metadata_file = contents.with_name(f'{contents.name}.tag')
        if metadata_file.exists():
            metadata = json.load(metadata_file.open('r'))
        else:
            metadata = {}

        return load_func(contents, **metadata)

    def locate_deep_key(self, key):
        return self._locate_deep_key(key, self)

    def _locate_deep_key(self, key, level, stack=()):
        locations = []
        for k, v in level.items():
            if k == key:
                locations.append((*stack, k))
            elif hasattr(v, 'items'):
                locations.extend(self._locate_deep_key(key, v, (*stack, k)))

        return locations

    def plot_reward_cumsum(self):
        rewards_dict = {}
        for eval_log in self.evaluate_logs:
            log = self[eval_log].log
            rewards_dict[eval_log[1]] = log[('balance', '0', 'reward')].cumsum()

        rewards = -1 * pd.DataFrame(rewards_dict)
        rewards /= rewards['modelpredictivecontrol'].values.reshape(-1, 1)
        rewards = rewards.iloc[10:]
        rewards = rewards.unstack().reset_index(name='Cumulative Cost')
        rewards = rewards.rename(columns={'level_0': 'Experiment', 'level_1': 'Step'})

        g = sns.lineplot(data=rewards, x='Step', y='Cumulative Cost', hue='Experiment')
        plt.show()

        print('here')


if __name__ == '__main__':
    rl = ResultLoader('/Users/ahalev/Dropbox/Avishai/gradSchool/internships/totalInternship/GridRL-V2/local/paper_experiments/mpc/experiments/experiment_logs')
    rl.plot_reward_cumsum()
    print('done')
