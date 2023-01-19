import argparse
import pandas as pd
import yaml


from pathlib import Path
from collections import UserDict
from warnings import warn

from config import Namespacify, nested_dict_update


class Config(Namespacify):
    def __init__(self, config=None, _default=False):
        self.default_config = DefaultConfig().with_name_from_keys('algo', 'type', prefix='Grid')

        super().__init__(self._parse_config())

        if config is not None:
            self._update_with_config(config)

        self.with_name_from_keys('algo', 'type', prefix='Grid')
        self._verbose(self.context.verbose)

    def _verbose(self, level):
        if level >= 2:
            print('Trainer config:')
            self.pprint(indent=1)
        elif level >= 1:
            print('Custom trainer config:')
            (self ^ self.default_config).pprint(indent=1)

    def _update_with_config(self, config, updatee=None):
        if isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)

        if updatee:
            nested_dict_update(updatee, config)
        else:
            nested_dict_update(self, config)

    def _get_arguments(self, key='', d=None):
        if d is None:
            d = self.default_config

        args = {}

        for k, v in d.items():
            new_key = f'{key}.{k}' if key else k
            if isinstance(v, (dict, UserDict)):
                args.update(self._get_arguments(key=new_key, d=v))
            else:
                args[new_key] = self._collect_argument(v)

        return args

    def _collect_argument(self, default_val):
        arg = {
            'default': default_val,
            'type': type(default_val),
        }
        if hasattr(default_val, '__len__') and not isinstance(default_val, str):
            arg["nargs"] = '+'

        return arg

    def _parse_config(self):
        parsed_args = self._create_parser().parse_known_args()

        if len(parsed_args[1]):
            bad_args = [x.replace("--", "") for x in parsed_args[1] if x.startswith("--")]
            valid_args = "\n\t\t".join(sorted(parsed_args[0].__dict__.keys()))
            warn(f'Unrecognized arguments {bad_args}.\n\tValid arguments:\n\t\t{valid_args}')

        config_file = parsed_args[0].__dict__.pop('config')
        restructured = self._restructure_arguments(parsed_args[0].__dict__)

        if config_file is not None:
            self._update_with_config(config_file, updatee=restructured)

        self._check_restructured(restructured, self.default_config)
        return restructured

    def _create_parser(self):
        parser = argparse.ArgumentParser(prog='GridRL')
        for arg_name, arg_info in self._get_arguments().items():
            parser.add_argument(f'--{arg_name}', **arg_info)

        parser.add_argument('--config', default=None)

        return parser

    def _restructure_arguments(self, config):
        if isinstance(config, dict):
            keys = [key.split('.') for key in config.keys()]
            keys_series = pd.Series(index=pd.MultiIndex.from_frame(pd.DataFrame(keys)), data=config.values())
        else:
            keys_series = config

        restructured = {}

        for key in keys_series.index.get_level_values(0).unique():
            subset = keys_series[key]
            if subset.index.dropna('all').empty:
                restructured.update({key: subset.item()})
            elif subset.index.dropna().nlevels == 1:
                restructured.update({key: subset.to_dict()})
            else:
                restructured.update({key: self._restructure_arguments(subset)})

        return restructured

    def _check_restructured(self, restructured, default_config, *stack):
        for key, value in default_config.items():
            if key not in restructured:
                raise RuntimeError(f'Missing key {"->".join([*stack, key])} in restructured config.')
            elif isinstance(value, dict):
                self._check_restructured(restructured[key], value, *stack, key)


class DefaultConfig(Namespacify):
    def __init__(self):
        super().__init__(self._load_default_config())

    def _load_default_config(self):
        contents = (Path(__file__).parent / 'default_config.yaml').open('r')
        return yaml.safe_load(contents)
