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

from microgrid_loader import microgrid_from_config


pymgrid.add_pymgrid_yaml_representers()


class ResultLoader(Namespacify):
    def __init__(self, result_dir, relevant_results=None, save_dir=None):
        super().__init__(self._load_results(Path(result_dir), relevant_vals=relevant_results))

        self.passed_result_dir = Path(result_dir)
        self.save_dir = Path(save_dir) if save_dir else None
        self.evaluate_logs = self.locate_deep_key('evaluate_log')
        self.configs = self.get_deep_values('config')
        self.microgrids = self._load_microgrids()
        self.log_columns = self.get_all_log_columns()

    def _load_results(self, directory, relevant_vals=None):

        results = {}
        for contents in directory.iterdir():
            if contents.is_dir():
                inner_res = self._load_results(contents, relevant_vals=relevant_vals)
                if len(inner_res):
                    print(f'Loaded results from: {contents}')
                    results[contents.name] = inner_res
                continue
            else:
                if not self.is_relevant(contents, relevant_vals):
                    continue

            if contents.suffix == '.yaml':
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

    def _load_microgrids(self):
        microgrid_locations = self.locate_deep_key('microgrid')
        microgrid_locations = [loc for loc in microgrid_locations if loc[-2] == 'config']

        microgrids = {
            loc[:-3]: microgrid_from_config(self[loc]) for loc in microgrid_locations
        }

        for loc, microgrid in microgrids.items():
            self[loc]['microgrid'] = microgrid

        return microgrids

    def get_deep_values(self, key):
        return [self[x] for x in self.locate_deep_key(key)]

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

    def get_all_log_columns(self):
        cols = pd.Index([])
        for eval_log in self.evaluate_logs:
            cols = cols.intersection(self[eval_log].log.columns)

        return cols

    def get_value_from_logs(self, log_column):
        rewards_dict = {}
        for eval_log in self.evaluate_logs:
            log = self[eval_log].log
            rewards_dict[eval_log[:-1]] = log[log_column]

        df = pd.concat(rewards_dict, axis=1)
        df.index.name = 'Step'

        return df

    def _save_file(self, suffix):
        if self.save_dir is not None:
            file = self.save_dir / suffix

            if not file.parent.exists():
                file.parent.mkdir(parents=True)

            print(f'Saving to file:\n\t{file.absolute()}')
            return file

        return None

    def plot_reward_cumsum(self, relative_to=None, hue=None, style=None, relplot_col=None, save=True):

        cost_column_name = f'{"Relative "*(relative_to is not None)}Cumulative Cost'

        cost = self.get_value_from_logs(('balance', '0', 'reward'))
        rewards = -1 * cost.cumsum()

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
        if save and save_file:
            plt.savefig(save_file)

        plt.show()

    def _extract_param_columns(self, df):
        new_df = {}
        for col in df.columns:
            split = df[col].str.split('_', expand=True)
            label = split.iloc[:, 0].unique().item().title()
            new_df[label] = pd.to_numeric(split.iloc[:, 1], errors='ignore')

        return pd.DataFrame(new_df, index=df.index)

    @classmethod
    def combine_loaders(cls, loaders_dict, relevant_results=None, save_dir=None):
        loaders_dict = cls.nested_dict_relevance(loaders_dict, relevant_results)
        return cls(loaders_dict, save_dir=save_dir)

    @classmethod
    def nested_dict_relevance(cls, nested_dict, relevant_results, parent_keys=()):
        # TODO something is wrong here it's only getting the first relevant value
        relevant = {}
        for k, v in nested_dict.items():
            if isinstance(v, (dict, UserDict)):
                inner_relevant = cls.nested_dict_relevance(v, relevant_results, (*parent_keys, k))
                if inner_relevant:
                    relevant[k] = inner_relevant
            elif cls.is_relevant((*parent_keys, k), relevant_results):
                relevant[k] = v

        return relevant

    @staticmethod
    def is_relevant(contents, relevant_vals):
        if relevant_vals is None:
            return True

        if isinstance(relevant_vals, str):
            relevant_vals = [relevant_vals]

        try:
            parts = contents.parts
        except AttributeError:
            parts = contents  # list-like

        for val in relevant_vals:
            if isinstance(val, list):
                list_match = all(ResultLoader.is_relevant(contents, inner_val) for inner_val in val)
                if list_match:
                    return True
            elif any(val in parts for val in relevant_vals):
                return True

        return False

    @staticmethod
    def log_without_forecasts(log, drop_singleton_level=False):
        df = log.drop(columns=log.columns[log.columns.get_level_values(-1).str.contains('forecast')])

        if drop_singleton_level:
            cols = df.columns
            df.columns = pd.MultiIndex.from_arrays([
                cols.get_level_values(j) for j in range(cols.nlevels) if cols.get_level_values(j).nunique() > 1])

        return df


if __name__ == '__main__':
    rl = ResultLoader('/Users/ahalev/Dropbox/Avishai/gradSchool/internships/totalInternship/GridRL-V2/local/paper_experiments/mpc/experiments/experiment_logs')
    rl.plot_reward_cumsum()
    print('done')
