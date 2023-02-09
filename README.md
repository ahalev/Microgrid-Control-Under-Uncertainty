## GridRL V2

The script `scripts/dqn_simulated_modular_env.py` is a garage script.

The script `scripts\dqn_simulated_env_rllib` is an RLLib script.


Even running in local mode, bugs in the environment that surface in RLLib
are a little more difficult to trace. Thus the garage script is best for env-related
bugs; it is possible the RLLib version provides better performance.

## Using `trainer.py`

`trainer.py` is a script made for reproducible experiments. It currently only supports experiments using `garage`.

The best way to use the trainer is to call it at the command line. By default, it will use the configuration 
settings defined in `config/default_config.yaml`.

```shell script
python trainer.py
```

You can pass custom settings and hyperparameters via the command line:

```shell script
python trainer.py --env.config.scenario 2 --context.experiment_name my_experiment
```

By default, `trainer.py` will store results in `./experiment_logs/dqn`. A `config.yaml` file will be saved 
in this directory, denoting the configuration used for the specific experiment. Experiment results will be saved here as well.

The location of results can be moved by modifying `context.log_dir` and `context.experiment_name`. 

### Additional methods of inputting custom hyperparameters

There are three other ways to define custom settings/hyperparameters:

  1. You can pass the `--config path_to_a_config.yaml` argument at the command line. 
  
     `path_to_a_config.yaml` may
     contain any combination of values as defined in `config/default_config.yaml`; they must be in the same format. 
     You may pass any number of config files this way:

     ```shell script
     --config path_to_a_config.yaml path_to_another_config.yaml
     ```

     If you pass multiple config files with conflicting values, the value from the ***last*** config file
     will be used.
     
     For example, suppose you have a yaml file `custom_config.yaml`:
     ```yaml
     microgrid:
        config:
          scenario: 1
     algo:
        sampler:
           type: local
           n_workers: 4 
     context:
        verbose: 1
     ```

     You can then use pass the path to your custom configuration file at the command line:
     ```shell script
     python trainer.py --config custom_config.yaml 
     ```
     and since `verbose=1`, this will print:

     ```text
     Custom trainer config:
     GridRL:
         microgrid:
             config:
                 scenario: 1
         algo:
             sampler:
                 type: local
                 n_workers: 4
         context:
             verbose: 1
     ```

     Note that any values passed this way will be overridden by explicit arguments or arguments passed by the below
     two methods.

  2. You can pass a path to a `yaml` file containing settings to `trainer.Trainer`. You may pass any combination
     of values as defined in `config/default_config.yaml`; they must be in the same format. This is equivalent to
     the above except it is done within a script and not at the command line:
     ```python
     from trainer import Trainer
     Trainer(config='custom_config.yaml').train()
     ```
  3. You can pass a nested dictionary defining configuration settings to `trainer.Trainer`.
     For example:
     ```python
     from trainer import Trainer
     config = {
         'microgrid': {'config': {'scenario': 1}},
         'algo': {'sampler': {'type': 'local', 'n_workers': 4}},
         'context': {'verbose': True}
     }
     Trainer(config=config).train()  
     ```
