import array
import gc
import logging
import multiprocessing
import os
import sys
import time
from typing import Iterable, Literal

# one dir lower than this script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- multiprocessing context ----
gc.freeze()  # docs suggest freezing to avoid copy-on-write
mp_ctx = multiprocessing.get_context("forkserver")

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
