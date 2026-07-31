import array
import gc
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
from typing import Iterable, Literal

# one dir lower than this script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- multiprocessing context ----
gc.freeze()  # docs suggest freezing to avoid copy-on-write
mp_ctx = multiprocessing.get_context("forkserver")

# ---- worker shutdown ----
# Seconds to wait for workers to exit before forcing the issue.
#
# The forkserver can leave exited workers unreaped, and the parent then blocks forever in
# join() on a sentinel that never fires. On this cluster it is not a rare race: it
# reproduced on a 200 KB corpus with 4 workers, and it cost 8.6 hours on one corpus build
# (16 workers, every one a zombie) and again on the BPE trainer in the same job (128
# zombies). In every case the parent already held all the results, so the hang discarded
# finished work.
#
# 60s rather than a longer wait because the workers have nothing left to do by the time
# either call site joins: both collect every result first. Waiting longer only adds to the
# stall when the condition fires, which here is most of the time.
WORKER_JOIN_TIMEOUT_S = 60

_shutdown_logger = logging.getLogger(__name__)


def _kill_forkserver(logger):
    """SIGKILL the forkserver, which is what actually releases a stalled join.

    The workers are already dead when this condition fires; they are zombies the
    forkserver never reaped, so signalling them changes nothing. Each worker's sentinel
    is a pipe from the forkserver, and killing the forkserver closes it. CPython handles
    exactly this: popen_forkserver.poll catches the resulting EOFError and sets returncode
    255, so every outstanding join returns at once. multiprocessing.forkserver
    .ensure_running reaps the dead forkserver and starts a new one on the next Process().
    """
    from multiprocessing import forkserver

    server = getattr(forkserver, "_forkserver", None)
    pid = getattr(server, "_forkserver_pid", None)
    if not pid:
        logger.warning("no forkserver to kill; workers may have used another start method")
        return False
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError) as e:
        logger.warning("could not kill forkserver %s: %s", pid, e)
        return False
    logger.warning("killed forkserver %s to release the stalled join", pid)
    return True


def _signal_all(processes, label, logger):
    """Send terminate or kill to every worker still alive, tolerating a dead pid."""
    for proc in processes:
        if not proc.is_alive():
            continue
        try:
            (proc.terminate if label == "terminate" else proc.kill)()
        except (ProcessLookupError, OSError, ValueError) as e:
            # Already reaped, or never started. Nothing to signal.
            logger.warning("worker %s could not be sent %s: %s", proc.pid, label, e)


def _await_all(processes, deadline):
    """Wait until every worker has exited or `deadline` passes; return those still alive."""
    for proc in processes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        proc.join(remaining)
    return [p for p in processes if p.is_alive()]


def join_workers(processes, logger=None, timeout=WORKER_JOIN_TIMEOUT_S):
    """Join worker processes, escalating to terminate and kill if the joins stall.

    `timeout` bounds each escalation stage across the whole set, not each worker: a
    stuck stage with 128 workers must not cost 128 * timeout. Returns the number of
    workers that never exited. Callers must already hold the workers' results; this
    reports a stuck worker loudly but does not raise, because aborting here would
    discard work that is already complete and correct.
    """
    logger = logger or _shutdown_logger
    processes = list(processes)
    alive = _await_all(processes, time.monotonic() + timeout)
    if alive:
        logger.warning("%d worker(s) did not exit within %ss", len(alive), timeout)
        _kill_forkserver(logger)
        alive = _await_all(alive, time.monotonic() + timeout)
    for label in ("terminate", "kill"):
        if not alive:
            return 0
        logger.warning("%d worker(s) still alive; sending %s", len(alive), label)
        _signal_all(alive, label, logger)
        alive = _await_all(alive, time.monotonic() + timeout)
    if alive:
        logger.error(
            "%d worker(s) still running after kill (pids %s); continuing without them",
            len(alive), ", ".join(str(p.pid) for p in alive),
        )
    return len(alive)


def shutdown_pool(pool, logger=None, timeout=WORKER_JOIN_TIMEOUT_S):
    """Close and join a multiprocessing Pool without risking an unbounded hang.

    Pool.join() takes no timeout, so it runs on a helper thread and the escalation
    happens here. As with `join_workers`, the caller must already hold every result.
    """
    logger = logger or _shutdown_logger
    pool.close()
    joiner = threading.Thread(target=pool.join, daemon=True)
    joiner.start()
    joiner.join(timeout)
    if not joiner.is_alive():
        return 0
    # Same cause and same remedy as join_workers: the pool's workers are unreaped
    # zombies, so killing the forkserver is what lets every join return.
    logger.warning("pool did not join within %ss", timeout)
    _kill_forkserver(logger)
    joiner.join(timeout)
    if not joiner.is_alive():
        return 0
    stuck = join_workers(list(getattr(pool, "_pool", ())), logger=logger, timeout=timeout)
    joiner.join(timeout)
    if joiner.is_alive():
        logger.error("pool join thread still blocked after killing workers; continuing")
    return stuck

# ---- typing ----

# Internal/output types
TokenSeq = array.array  # [int]
PretokenizedT = list[TokenSeq]

# inputs more flexible
InputTokenSeq = array.array | list[int]

# shared aliases
DigitHandlingT = Literal["RTL3", "SPLIT"] | None
TokenPairT = tuple[int, int]


def token_array(values: Iterable[int]) -> TokenSeq:
    return array.array("i", values)


# ---- logging ----


def create_logger(tag: str, verbose: bool = True):
    default_fields = logging.getLogRecordFactory()
    t0 = time.perf_counter()

    # https://stackoverflow.com/questions/63056270/python-logging-time-since-start-in-seconds
    def record_factory(*args, **kwargs):
        record = default_fields(*args, **kwargs)
        record.uptime = time.perf_counter() - t0
        record.level_nocaps = record.levelname.lower()
        return record

    logging.setLogRecordFactory(record_factory)
    logger = logging.getLogger(tag)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter(f"[%(uptime)6.1fs][{tag}] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
