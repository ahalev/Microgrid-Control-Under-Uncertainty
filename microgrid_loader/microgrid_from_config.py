import yaml

from pymgrid import Microgrid


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
        getattr(microgrid, method)(**method_params)


def _set_microgrid_attributes(microgrid, config):
    try:
        attributes = config.attributes
    except AttributeError:
        return

    for attr, value in attributes.items():
        if isinstance(value, str) and value.startswith('!'):
            try:
                value = yaml.safe_load(value)
            except yaml.YAMLError:
                value = yaml.safe_load(f'{value} {{}}')

        setattr(microgrid, attr, value)
