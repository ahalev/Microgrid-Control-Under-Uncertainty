import numpy as np
import pandas as pd
import pymgrid
import seaborn as sns
import json
import yaml
import warnings

from collections import UserDict
from expfig import Namespacify
from matplotlib import pyplot as plt
from pathlib import Path

from microgrid_loader import microgrid_from_config


pymgrid.add_pymgrid_yaml_representers()


class ResultLoader(Namespacify):
    """

    Parameters
    ----------
    result_dir : str or Path
        Directory to load results from.

    relevant_results : list[Union[str, list[str]]] or None, default None
        Results to collect. A list of strings to match as subdirectories in result_dir.
        If an element is a list, it will only collect results that match every element in that list.

        Example:
        >>> relevant_results = ['scenario_0', ['scenario_1', 'forecaster_0.0']]
        Will load results that match either 'scenario_0' or 'scenario_1' AND 'forecaster_0.0'.

    save_dir : str, Path or None, default None
        Location to save figures and other results. If None, do not save other results.
    """

    def __init__(self, results_or_dir, relevant_results=None, save_dir=None):

        results, result_dir = self._get_dict_results(results_or_dir, relevant_results)
        super().__init__(results)

        self.passed_result_dir = result_dir
        self.save_dir = Path(save_dir) if save_dir else None
        self.evaluate_logs = self.locate_deep_key('evaluate_log')
        self.configs = self.get_deep_values('config')
        self.microgrids = self._load_microgrids()
        self.log_columns = self.get_all_log_columns()

    def _get_dict_results(self, results_or_dir, relevant_results):
        if isinstance(results_or_dir, dict):
            if relevant_results:
                warnings.warn('Non-empty relevant_results will be ignored when combining results.')
            return results_or_dir, None

        result_dir = Path(results_or_dir)
        return self._load_results(result_dir, relevant_vals=relevant_results), result_dir

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
            elif isinstance(v, (dict, UserDict)):
                locations.extend(self._locate_deep_key(key, v, (*stack, k)))

        return locations

    def get_all_log_columns(self):
        cols = None
        for eval_log in self.evaluate_logs:
            if cols is None:
                cols = self[eval_log].log.columns
            else:
                cols = cols.intersection(self[eval_log].log.columns)

        if len(cols) == 0:
            raise RuntimeError

        return cols

    def get_value_from_logs(self, log_column):
        rewards_dict = {}
        for eval_log in self.evaluate_logs:
            log = self[eval_log].log
            rewards_dict[eval_log[:-1]] = log.loc[:, log_column]

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

    def plot_forecast_vs_true(
            self,
            module_name,
            module_kind,
            max_forecast_steps=None,
            step_range=slice(None),
            relative=False
    ):
        # Looks for forecast of the type f'{module_kind}_forecast_0, f'{module_kind}_forecast_1, etc
        forecast_horizon = [c.config.microgrid.methods.set_forecaster.forecast_horizon for c in self.configs]
        max_forecast_horizon = max(forecast_horizon)

        applicable_cols = [f'{module_kind}_forecast_{j}' for j in range(max_forecast_horizon)]
        forecasts = self.get_value_from_logs([(module_kind, '0', col) for col in applicable_cols])
        true = self.get_value_from_logs((module_kind, '0', f'{module_kind}_current'))

        forecasts = forecasts.loc[step_range]
        forecasts = forecasts.unstack(level=-1).reset_index()
        forecasts = self._extract_param_columns(forecasts)
        forecasts = forecasts.rename(columns={0: module_kind.title()})

        forecasts['Forecasted Step'] = forecasts['Step'] + forecasts['Load Forecast'] + 1

        true = true.unstack(level=-1).reset_index()
        true = self._extract_param_columns(true)
        true = true.rename(columns={0: module_kind.title()})

        true['Forecasted Step'] = 'True'

        if relative:
            forecasts = self._relative_forecasts(forecasts, true, module_kind)

        if max_forecast_steps is not None:
            forecasts = forecasts[forecasts['Load Forecast'] <= max_forecast_steps]

        true = true[true['Step'].isin(forecasts['Forecasted Step'])]

        grid = sns.relplot(
            data=forecasts,
            x='Forecasted Step',
            y=module_kind.title(),
            col='Scenario',
            row='Forecaster',
            hue='Step',
            kind='line'
        )

        if not relative:
            for ax in grid.axes_dict.values():
                sns.lineplot(
                    data=true,
                    x='Step',
                    y=module_kind.title(),
                    hue='Forecasted Step',
                    ax=ax,
                    linewidth=2
                )

        plt.show()

    def _relative_forecasts(self, forecasts, true, module_kind):
        forecast_subset = forecasts[['Forecasted Step', module_kind.title()]]
        forecast_subset = forecast_subset.set_index(['Forecasted Step'], append=True)

        true_subset = true[['Step', module_kind.title()]].set_index('Step').drop_duplicates()

        difference = forecast_subset.subtract(true_subset, level='Forecasted Step')
        relative = difference.div(true_subset, level='Forecasted Step')

        forecasts['Load'] = relative.droplevel(level=1)
        return forecasts

    def plot_reward_cumsum(self,
                           *,
                           module=None,
                           relative_to=None,
                           transient_steps=100,
                           hue=None,
                           style=None,
                           units=None,
                           relplot_col=None,
                           save=True
                           ):

        cost_column_name = f'{"Relative "*(relative_to is not None)}Cumulative Cost'

        module = module if module else 'balance'

        cost = self.get_value_from_logs((module, '0', 'reward'))
        rewards = -1 * cost.cumsum()

        rewards = self._make_relative(rewards, relative_to)

        param_cols = [f'level_{j}' for j in range(len(self.evaluate_logs[0][:-1]))]

        rewards = rewards.iloc[transient_steps:]
        rewards = rewards.unstack().reset_index(name=cost_column_name)

        rewards = pd.concat([rewards.drop(columns=param_cols), self._extract_param_columns(rewards[param_cols])], axis=1)

        rewards = rewards.rename(columns={f'level_{len(self.evaluate_logs[0])-1}': 'Step'})

        min_col_wrap, max_col_wrap = 2, 5

        if relplot_col is not None:
            nunique_col = rewards[relplot_col].nunique()
            col_wrap = nunique_col / np.arange(min_col_wrap, max_col_wrap+1)
            col_wrap = np.where(col_wrap > 2)[0]
            try:
                col_wrap = col_wrap.max() + min_col_wrap
            except ValueError:
                col_wrap = None
        else:
            col_wrap = int((min_col_wrap+max_col_wrap) // 2)

        g = sns.relplot(
            data=rewards,
            x='Step',
            y=cost_column_name,
            kind='line',
            hue=hue,
            style=style,
            units=units,
            col=relplot_col,
            col_wrap=col_wrap,
            palette=sns.color_palette("rocket_r", n_colors=rewards[hue].nunique()),
            estimator='mean' if units is None else None
        )

        if relative_to is not None:
            g.set(ylim=(0.75, 1.25))

        for ax in g.axes_dict.values():
            ax.set_title(f'{ax.get_title()} ({module.title()} Cost)')

        if save:
            save_file = self._save_file('reward_cumsum.png')
            if save_file:
                plt.savefig(save_file)

        plt.show()

    def _make_relative(self, rewards, relative_to):
        if relative_to is None:
            return rewards

        if isinstance(relative_to, str):
            relative_to = [relative_to]

        relative_to_loc = {j: rel_to
                           for j, level in enumerate(rewards.columns.levels)
                           for rel_to in relative_to if rel_to in level
                           }

        idx = tuple(relative_to_loc[j] if j in relative_to_loc else slice(None) for j in range(rewards.columns.nlevels))
        values_relative_to = rewards.loc[:, idx]
        values_relative_to = values_relative_to.droplevel(list(relative_to_loc.keys()), axis=1)

        # for col in values_relative_to.columns:
        for col in rewards.columns:
            relative_slice = tuple(c for j, c in enumerate(col) if j not in relative_to_loc.keys())
            rewards[col] = rewards[col].div(values_relative_to[relative_slice], axis=0)

        return rewards

    def _extract_param_columns(self, df):
        new_df = {}
        for col in df.columns:
            try:
                contains_underscore = df[col].str.contains('_').all()
            except AttributeError:
                new_df[col] = df[col]
                continue

            if contains_underscore:
                split = df[col].str.split('_', expand=True)
                label = split.iloc[:, :-1].value_counts().index.item()
                label = ' '.join(x.title() for x in label)
                new_df[label] = pd.to_numeric(split.iloc[:, -1], errors='ignore')
            else:
                new_df[col] = df[col]

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
