import argparse
import pandas as pd
import yaml


from pathlib import Path
from config import Namespacify
from pymgrid import envs

BuiltinFunctionType = type(len)


def is_builtin_class_instance(obj):
    return obj.__class__.__module__ == 'builtins'


class Config(Namespacify):
    def __init__(self):
        self.default_config = self._load_default_config()
        super().__init__('GridRL', self._parse_config())

    def _load_default_config(self):
        contents = (Path(__file__).parent / 'default_config.yaml').open('r')
        return yaml.safe_load(contents)

    def _get_arguments(self, key='', d=None):
        if d is None:
            d = self.default_config

        args = {}

        for k, v in d.items():
            new_key = f'{key}.{k}' if key else k
            if isinstance(v, dict):
                args.update(self._get_arguments(key=new_key, d=v))
            else:
                args[new_key] = self._collect_argument(v)

        return args

    def _collect_argument(self, default_val):
        arg = {
            'default': default_val,
            'type': type(default_val),
        }
        # if isinstance(default_val, bool):
        #     arg['action'] = 'store_true'
        #
        if hasattr(default_val, '__len__') and not isinstance(default_val, str):
            arg["nargs"] = '+'

        return arg

    def _parse_config(self):
        parsed_args = self._create_parser().parse_known_args()
        restructured = self._restructure_arguments(parsed_args[0].__dict__)
        self._check_restructured(restructured, self.default_config)
        return restructured

    def _create_parser(self):
        parser = argparse.ArgumentParser(prog='GridRL')
        for arg_name, arg in self._get_arguments().items():
            parser.add_argument(f'--{arg_name}', **arg)

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


if __name__ == '__main__':
    c = Config()
    # print(Namespacify('config', c.default_config))
    print(c._get_arguments())
