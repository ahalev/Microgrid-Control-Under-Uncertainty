import contextlib
import queue
import subprocess
import time
import threading

from expfig import get_logger

logger = get_logger()


@contextlib.contextmanager
def run_and_terminate_process(*args, **kwargs):
    """
    Open a new subprocess and kill it if parent receives a SIGINT or SIGTERM.

    Note that child process will survive if parent receives a SIGKILL.

    :param args:
    :param kwargs:
    :return:
    """
    p = subprocess.Popen(*args, **kwargs)
    logger.info(f'Launched child with pid: {p}')
    logger.info(f'Child command: {" ".join(p.args)}')

    try:
        yield p
    finally:
        if p.poll() is None:
            print(f'Killing process due to uncaught exception:')
            print('\t', ' '.join(p.args))

            p.terminate()  # send sigterm, or ...
            p.kill()  # send sigkill


def kill_hanging(p, timeout=3600, sleep_interval=None):

    if sleep_interval is None:
        sleep_interval = timeout // 60

    q = queue.Queue()
    t = threading.Thread(target=_enqueue_output, args=(p.stdout, q), daemon=True)
    t.start()

    sleep_secs = 0

    while sleep_secs < timeout:
        try:
            line = q.get_nowait()
        except queue.Empty:
            if p.poll() is not None:
                # process finished
                finish(p)
                break

            time.sleep(sleep_interval)
            sleep_secs += sleep_interval
        else:
            if not line and p.poll() is not None:
                # process finished
                finish(p)
                break

            print(line, sep='')
            sleep_secs = 0
    else:
        msg = f'Killing process due to no output after {sleep_secs}>{timeout} seconds:\n\t{" ".join(p.args)}'
        p.terminate()
        p.kill()

        logger.info(msg)


def _enqueue_output(out, queue_obj):
    for line in iter(out.readline, b''):
        queue_obj.put(line.rstrip())

    out.close()


def finish(p):
    proc_str = " ".join(p.args)
    logger.info(f'process "{proc_str}" completed')
