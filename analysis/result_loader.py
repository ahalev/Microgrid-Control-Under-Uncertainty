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

    def _save_file(self, suffix):
        if self.save_dir is not None:
            file = self.save_dir / suffix

            if not file.parent.exists():
                file.parent.mkdir(parents=True)

            print(f'Saving to file:\n\t{file.absolute()}')
            return file

        return None

    def plot_reward_cumsum(self, relative_to=None, hue=None, style=None, relplot_col=None):
        rewards_dict = {}
        for eval_log in self.evaluate_logs:
            log = self[eval_log].log
            rewards_dict[eval_log[:-1]] = log[('balance', '0', 'reward')].cumsum()

        cost_column_name = f'{"Relative "*(relative_to is not None)}Cumulative Cost'

        rewards = -1 * pd.DataFrame(rewards_dict)
        if relative_to is not None:
            relative_to_loc = [relative_to in level for level in rewards.columns.levels]
            idx = tuple(slice(None) if not loc else relative_to for loc in relative_to_loc)
            values_relative_to = rewards.loc[:, idx]
            values_relative_to = values_relative_to.droplevel(np.where(relative_to_loc)[0].item(), axis=1)

            for col in values_relative_to.columns:
                rewards[col] = rewards[col].div(values_relative_to[col], axis=0)

        param_cols = [f'level_{j}' for j in range(len(self.evaluate_logs[0][:-1]))]

        rewards = rewards.iloc[10:]
        rewards = rewards.unstack().reset_index(name=cost_column_name)

        rewards = pd.concat([rewards.drop(columns=param_cols), self._extract_param_columns(rewards[param_cols])], axis=1)

        rewards = rewards.rename(columns={f'level_{len(self.evaluate_logs[0])-1}': 'Step'})

        min_col_wrap, max_col_wrap = 2, 5

        if relplot_col is not None:
            nunique_col = rewards[relplot_col].nunique()
            col_wrap = nunique_col / np.arange(min_col_wrap, max_col_wrap+1)
            col_wrap = np.where(col_wrap > 2)[0].max() + min_col_wrap
        else:
            col_wrap = int((min_col_wrap+max_col_wrap) // 2)

        g = sns.relplot(
            data=rewards,
            x='Step',
            y=cost_column_name,
            kind='line',
            hue=hue,
            style=style,
            col=relplot_col,
            col_wrap=col_wrap,
            palette=sns.color_palette("rocket_r", n_colors=rewards[hue].nunique()))

        if relative_to is not None:
            g.set(ylim=(0.5, 1.5))

        save_file = self._save_file('reward_cumsum.png')
        if save_file:
            plt.savefig(save_file)

        plt.show()

    def _extract_param_columns(self, df):
        new_df = {}
        for col in df.columns:
            split = df[col].str.split('_', expand=True)
            label = split.iloc[:, 0].unique().item().title()
            new_df[label] = pd.to_numeric(split.iloc[:, 1], errors='ignore')

        return pd.DataFrame(new_df, index=df.index)


if __name__ == '__main__':
    rl = ResultLoader('/Users/ahalev/Dropbox/Avishai/gradSchool/internships/totalInternship/GridRL-V2/local/paper_experiments/mpc/experiments/experiment_logs')
    rl.plot_reward_cumsum()
    print('done')
