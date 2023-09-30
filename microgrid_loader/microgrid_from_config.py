import yaml

from functools import reduce
from pymgrid import Microgrid


CMD_LINE_YAML_REPLACEMENTS = [('(', ' '), (')', ' '), (':', ': ')]


def microgrid_from_config(microgrid_config):
    if isinstance(microgrid_config.config, Microgrid):
        microgrid = microgrid_config.config
    else:
        microgrid_yaml = f'!Microgrid\n{yaml.safe_dump(microgrid_config.config.data)}'
        try:
            microgrid = yaml.safe_load(microgrid_yaml)
        except yaml.YAMLError:
            raise yaml.YAMLError(f'Unable to parse microgrid yaml:\n{microgrid_yaml}')

    _post_process_microgrid(microgrid, microgrid_config)
    return microgrid


def _post_process_microgrid(microgrid, config):
    _call_microgrid_methods(microgrid, config)
    _set_microgrid_attributes(microgrid, config)


def _call_microgrid_methods(microgrid, config):
    try:
        methods = config.methods
    except AttributeError:
        return

    for method, method_params in methods.items():
        _params = {k: _check_if_yaml(v) for k, v in method_params.items()}
        getattr(microgrid, method)(**_params)


def _set_microgrid_attributes(microgrid, config):
    try:
        attributes = config.attributes
    except AttributeError:
        return

    for attr, value in attributes.items():
        value = _check_if_yaml(value)

        setattr(microgrid, attr, value)


def _check_if_yaml(value):
    if isinstance(value, str) and value.startswith('!'):
        return _load_yaml_value(value)

    return value

def _load_yaml_value(value):
    try:
        value = yaml.safe_load(value)
    except yaml.YAMLError:
        try:
            value = yaml.safe_load(f'{value} {{}}')
        except yaml.YAMLError:
            value = reduce(lambda _str, kv: _str.replace(*kv), CMD_LINE_YAML_REPLACEMENTS, value)
            value = yaml.safe_load(value)

    return value
