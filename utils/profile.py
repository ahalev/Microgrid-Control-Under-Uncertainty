import contextlib
import cProfile
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
