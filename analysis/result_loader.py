import numpy as np
import pandas as pd
import pymgrid
import re
import seaborn as sns
import json
import yaml
import warnings

from collections import UserDict
from expfig import Namespacify
from functools import reduce
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

    renamer : dict or None, default None
        Rename keys in loaded results.

    replacements : list of callable, default ()
        Replacements of the keys. Takes in a key and and returns a new key. For example:
        >>> replacements = [lambda x: x.replace('scenario', 'sro')]

        will replace all keys of the form 'scenario<stuff>' with 'sro<stuff>'

        .. warning::
            This will be done to ALL keys. Be careful, easy to get bitten.

    save_dir : str, Path or None, default None
        Location to save figures and other results. If None, do not save other results.
    """

    def __init__(self,
                 results_or_dir,
                 relevant_results=None,
                 save_dir=None,
                 renamer=None,
                 replacements=(),
                 verbose=1):
        self.verbose = verbose

        results, result_dir = self._get_dict_results(results_or_dir, relevant_results, replacements)
        super().__init__(results)
        self._rename(renamer)
        self._check_renamed(renamer)

        self.result_dir = result_dir
        self.save_dir = Path(save_dir) if save_dir else None

        self.result_list = self._get_result_list()
        self.microgrids = self._load_microgrids()
        self.log_columns = self.get_all_log_columns()

    def _get_dict_results(self, results_or_dir, relevant_results, replacements):
        if isinstance(results_or_dir, (dict, UserDict)):
            if relevant_results:
                warnings.warn('Non-empty relevant_results will be ignored when combining results.')
            if replacements:
                warnings.warn('Non-empty replacements will be ignored when combining results.')
            return results_or_dir, {name: results.result_dir for name, results in results_or_dir.items()}

        result_dir = Path(results_or_dir)
        replacement_func = lambda x: reduce(lambda string, f: f(string), replacements, x)

        return self._load_results(result_dir, relevant_results, replacement_func), result_dir

    def _load_results(self, directory, relevant_vals, replacement_func):
        results = {}
        for contents in directory.iterdir():
            if contents.is_dir():
                inner_res = self._load_results(contents, relevant_vals, replacement_func)
                if len(inner_res):
                    if self.verbose and any(k in inner_res.keys() for k in ('train_log', 'evaluate_log')):
                        print(f'Loaded results from: {contents}')
                    results[replacement_func(contents.name)] = inner_res
                continue
            else:
                if not self.is_relevant(contents, relevant_vals):
                    if self.verbose >= 2:
                        print(f'Ignoring non-relevant contents:\n\t{contents}')
                    continue

            results.update(read_file(replacement_func(contents.stem), contents, self.verbose))

        return results

    def _rename(self, renamer, lvl=None):
        if renamer is None:
            return

        if lvl is None:
            lvl = self

        for k, v in list(lvl.items()).copy():
            if k in renamer.keys():
                lvl.update({renamer[k]: lvl.pop(k)})
                v = lvl[renamer[k]]

            if isinstance(v, Namespacify):
                self._rename(renamer, lvl=v)

        # for old_key, new_key in renamer.items():
        #     if old_key in

        # self.update()

    def _check_renamed(self, renamer):
        if renamer is None:
            return
        for k in renamer:
            try:
                existing_key_locs =  self.locate_deep_key(k)
            except AttributeError:
                pass
            else:
                if existing_key_locs:
                    raise RuntimeError

    def _load_microgrids(self):
        microgrid_locations = self.locate_deep_key('microgrid')
        microgrid_locations = [loc for loc in microgrid_locations if loc[-2] == 'config']

        microgrids = {
            loc[:-3]: microgrid_from_config(self[loc]) for loc in microgrid_locations
        }

        for loc, microgrid in microgrids.items():
            self[loc]['microgrid'] = microgrid

        return microgrids

    def _get_result_list(self):
        return [x[:-1] for x in self.locate_deep_key('evaluate_log')]

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
        for result in self.iterlist():
            if cols is None:
                cols = result.evaluate_log.log.columns
            else:
                to_add = result.evaluate_log.log.columns
                try:
                    cols = cols.intersection(to_add)
                except ValueError:
                    if cols.nlevels == 3:
                        assert len(cols.get_level_values(1).unique()) == 1  # column level of '0', can be dropped
                        cols = cols.droplevel(level=1)
                    elif to_add.nlevels == 3:
                        assert len(to_add.get_level_values(1).unique()) == 1  # column level of '0', can be dropped
                        to_add = to_add.droplevel(level=1)
                    else:
                        raise

                    cols = cols.intersection(to_add)

        if cols is None:
            raise RuntimeError(f'No results found in self.result_list in result dir:\n\t{self.result_dir}.')

        if len(cols) == 0:
            raise RuntimeError

        return cols

    def get_value_from_logs(self, log_column,
                            fill_missing_levels=True,
                            missing_level_fill_val='MISSING',
                            errors='ignore'):
        rewards_dict = {}
        for loc, result in self.iterdict():
            log = result.evaluate_log.log
            try:
                rewards_dict[loc] = log.loc[:, log_column]
            except KeyError:
                try:
                    rewards_dict[loc] = log.loc[:, (log_column[0], log_column[-1])]
                except KeyError:
                    if errors == 'ignore':
                        pass
                    elif errors == 'raise':
                        msg = f'Evaluate log at position below does not contain key.\n\tPosition: {loc}\n\tkey: {log_column}'
                        raise KeyError(msg)
                    else:
                        raise ValueError(f"Unrecognized error handling '{errors}'")

        if fill_missing_levels:
            rewards_dict = self._fill_missing_levels(rewards_dict, missing_level_fill_val)

        df = pd.concat(rewards_dict, axis=1)
        df.index.name = 'Step'

        return df

    def _fill_missing_levels(self, rewards_dict, missing_level_fill_val):
        key_groups = []

        def get_group(key):
            partition = key.rpartition('_')
            if not partition[1]:
                raise TypeError(f"Key '{key}' is missing delimiter '_' and cannot be split properly. ")

            return partition[0], partition[2]

        for key in rewards_dict.keys():
            key_group = dict([get_group(k) for k in key])
            key_groups.append(key_group)

        group_frame = pd.DataFrame(key_groups)
        group_frame = group_frame.fillna(missing_level_fill_val)

        filled_key_groups = group_frame.to_dict(orient='records')
        filled_keys = [tuple(f'{k}_{v}' for k, v in group.items()) for group in filled_key_groups]

        for old_key, filled_key in zip(rewards_dict.keys(), filled_keys):
            assert all(k in filled_key for k in old_key)

        rewards_dict = dict(zip(filled_keys, rewards_dict.values()))

        return rewards_dict

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
        forecast_horizon = [c.config.config.microgrid.methods.set_forecaster.forecast_horizon for c in self.iterlist()]
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
                           palette=None,
                           markers=None,
                           relplot_col=None,
                           save=True,
                           facet_kws=None,
                           **kwargs):
        module = module if module else 'balance'
        y = (module, '0', 'reward')

        return self.plot(
            x=None,
            y=y,
            op=lambda x: -1 * x,
            cumulative=True,
            relative_to=relative_to,
            transient_steps=transient_steps,
            kind='line',
            hue=hue,
            style=style,
            units=units,
            palette=palette,
            markers=markers,
            relplot_col=relplot_col,
            ylabel=f'{"Relative "*(relative_to is not None)}Cumulative Cost',
            save=save,
            facet_kws=facet_kws,
            **kwargs,
        )

    def plot(self,
             *,
             x=None,
             y=None,
             op=None,
             cumulative=False,
             relative_to=None,
             transient_steps=100,
             kind='line',
             hue=None,
             style=None,
             units=None,
             palette=None,
             markers=None,
             relplot_col=None,
             xlabel=None,
             ylabel=None,
             save=True,
             facet_kws=None,
             **kwargs):

        if x is None:
            x = 'Step'
            xval = None
            if xlabel is None:
                xlabel = 'Step'
        else:
            xval = self.get_value_from_logs(x)
            if xlabel is None:
                xlabel = x[-1].title()

        if y is None:
            raise ValueError("Variable 'y' cannot be None")

        if ylabel is None:
            ylabel = y[-1].title()
            if cumulative:
                ylabel = f'Cumulative {ylabel}'
            if relative_to is not None:
                ylabel = f'Relative {ylabel}'

        yval = self.get_value_from_logs(y)

        if op is not None:
            yval = op(yval)

        if cumulative:
            yval = yval.cumsum()

        yval = self._make_relative(yval, relative_to)

        yval = yval.iloc[transient_steps:]
        yval = yval.unstack()

        if xval is not None:
            xval = xval.iloc[transient_steps:]
            xval = xval.unstack()
            yval = pd.concat({xlabel: xval, ylabel: yval}, axis=1)
            yval = yval.reset_index()
        else:
            yval = yval.reset_index(name=ylabel)

        param_cols = yval.columns[yval.columns.str.contains('level_')]
        yval = pd.concat([yval.drop(columns=param_cols), self._extract_param_columns(yval[param_cols])], axis=1)

        min_col_wrap, max_col_wrap = 2, 5

        if relplot_col is not None:
            nunique_col = yval[relplot_col].nunique()
            col_wrap = nunique_col / np.arange(min_col_wrap, max_col_wrap+1)
            col_wrap = np.where(col_wrap > 2)[0]
            try:
                col_wrap = col_wrap.max() + min_col_wrap
            except ValueError:
                col_wrap = None
        else:
            col_wrap = None

        kwargs = {'estimator': 'mean', **kwargs}

        if kind == 'scatter':
            if units:
                warnings.warn("Ignoring 'units' kwarg with kind='scatter'.")
                units = None

            kwargs.pop('estimator')

        elif units:
            kwargs['estimator'] = None

        g = sns.relplot(
            data=yval,
            x=xlabel,
            y=ylabel,
            kind=kind,
            hue=hue,
            style=style,
            units=units,
            markers=markers,
            col=relplot_col,
            col_wrap=col_wrap,
            palette=sns.color_palette(palette, n_colors=yval[hue].nunique()) if hue is not None else None,
            facet_kws=facet_kws,
            **kwargs
        )

        for ax in g.axes_dict.values():
            ax.set_title(f'{ax.get_title()} ({y[0].title()} {y[-1]})')

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

        for col in rewards.columns:
            rewards[col] = self._get_relative_column(rewards[col], values_relative_to, relative_to_loc.keys())

        return rewards

    def _get_relative_column(self, original_col, values_relative_to, levels_with_relative_vals):
        relative_slice = tuple(c for j, c in enumerate(original_col.name) if j not in levels_with_relative_vals)
        try:
            relative_col = original_col.div(values_relative_to[relative_slice], axis=0)
        except KeyError:
            _relative_slice = list(relative_slice)
            missing = []

            for lvl, value in enumerate(relative_slice):
                unique_in_lvl = values_relative_to.columns.get_level_values(lvl).unique()

                if len(unique_in_lvl) == 1:
                    _relative_slice[lvl] = unique_in_lvl.item()
                else:
                    missing.append((lvl, value, unique_in_lvl))

            if missing:
                nlnt2 = '\n\t\t'
                missing_msg = [f"Level: {j}{nlnt2}Missing or non-unique Value: {val}{nlnt2}Existing Values: {lvl_vals}"
                               for j, val, lvl_vals in missing]
                missing_msg = '\n\t'.join(missing_msg)

                msg = f'The combination {_relative_slice} does not exist.\nThe following values are missing:' \
                      f'\n\t{missing_msg}\n\n\t Do you need to make your result relative to some value in these column(s) ' \
                      f'as well?'
                raise ValueError(msg)

            relative_col = original_col.div(values_relative_to[tuple(_relative_slice)], axis=0)

        return relative_col

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
                label = split.iloc[:, :-1].value_counts().index
                try:
                    label = label.item()
                except ValueError:
                    nlnt = '\n\t'
                    raise ValueError(f'Clashing labels for parameter:{nlnt}{nlnt.join(label)}')

                label = ' '.join(x.title() for x in label)
                new_df[label] = pd.to_numeric(split.iloc[:, -1], errors='ignore')
            else:
                new_df[col] = df[col]

        return pd.DataFrame(new_df, index=df.index)

    @classmethod
    def combine_loaders(cls, loaders_dict, relevant_results=None, save_dir=None):
        relevant_loaders = cls.nested_dict_relevance(loaders_dict, relevant_results)
        return cls(relevant_loaders, save_dir=save_dir)

    @classmethod
    def nested_dict_relevance(cls, nested_dict, relevant_results, parent_keys=()):
        relevant = nested_dict.copy()
        relevant.clear()

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
        """

        Parameters
        ----------
        contents : Path or tuple
        relevant_vals : str or list-like

        Returns
        -------
        bool

        """
        if relevant_vals is None:
            return True

        if isinstance(relevant_vals, str):
            relevant_vals = [relevant_vals]

        try:
            parts = contents.parts
        except AttributeError:
            parts = contents  # list-like

        for val in relevant_vals:
            if pd.api.types.is_list_like(val):
                list_match = all(ResultLoader.is_relevant(contents, inner_val) for inner_val in val)
                if list_match:
                    return True
            elif val in parts:
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

    def absolute_result_paths(self):
        if isinstance(self.result_dir, (dict, UserDict)):
            return [self.result_dir[loc[0]].joinpath(*loc[1:]) for loc in self.result_list]

        return [self.result_dir.joinpath(*loc) for loc in self.result_list]

    def iterlist(self):
        for loc in self.result_list:
            yield self[loc]

    def iterdict(self):
        for loc in self.result_list:
            yield loc, self[loc]

    def pop(self, __key):
        super().pop(__key)
        self.result_list = self._get_result_list()
        self.microgrids = self._load_microgrids()

        try:
            self.result_dir.pop(__key)
        except AttributeError:
            pass

    @staticmethod
    def load_shallow_results(
            directory,
            relevant_results=None,
            renamer=None,
            replacements=()
    ):

        results = set()
        for contents in Path(directory).iterdir():
            # TODO you need to figure our what to do with relevant_vals and replacement func
            if contents.name in ('evaluate_log', 'train_log', 'config'):
                results.add(contents.parent)
                break
            elif contents.is_dir():
                results |= set(ResultLoader.load_shallow_results(contents, relevant_results, renamer, replacements))

        return sorted(results)


def read_file(contents_name, contents, verbose):
    loaded_contents = None

    if contents.suffix == '.yaml':
        loaded_contents = yaml.safe_load(contents.open('r'))
    elif contents.suffix in ('.csv', '.xlsx'):
        load_func = pd.read_csv if contents.suffix == '.csv' else pd.read_excel
        try:
            loaded_contents = read_pandas(contents, load_func)
        except ValueError as e:
            warnings.warn('Exception encountered when loading file into pandas DataFrame.\n\t'
                          f'File: {contents}\n\tException: {e.__class__.__name__}({e})')
    elif contents.suffix == '.tag':
        pass
    elif verbose >= 2:
        warnings.warn(f'Ignoring file of unrecognized type {contents.name}')

    if loaded_contents is not None:
        return {contents_name: loaded_contents}

    return dict()


def read_pandas(contents, load_func):
    metadata_file = contents.with_name(f'{contents.name}.tag')
    if metadata_file.exists():
        metadata = json.load(metadata_file.open('r'))
    else:
        metadata = {}

    return load_func(contents, **metadata)
