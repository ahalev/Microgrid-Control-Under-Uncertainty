## GridRL V2

The script `scripts/dqn_simulated_modular_env.py` is a garage script.

The script `scripts\dqn_simulated_env_rllib` is an RLLib script.


Even running in local mode, bugs in the environment that surface in RLLib
are a little more difficult to trace. Thus the garage script is best for env-related
bugs; it is possible the RLLib version provides better performance.
