import contextlib
import cProfile
import functools
import os
import pstats

from expfig import get_logger
from pathlib import Path

logger = get_logger()


@contextlib.contextmanager
def profile(profile_path):
    pr = cProfile.Profile()
    pr.enable()

    try:
        yield
    finally:
        pr.disable()
        stats = pstats.Stats(pr)
        stats.dump_stats(profile_path)
        logger.info(f'Profile available at: [{Path(profile_path).resolve()}]')


def profile_wrapper(func, profile_path):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if callable(profile_path):
            _prof_path = profile_path(*args, **kwargs)
        else:
            _prof_path = profile_path

        with profile(_prof_path):
            return func(*args, **kwargs)

    return wrapper


def garage_profile_path(parent_dir, itr, *_, **__):
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f'epoch_{itr}.prof')
