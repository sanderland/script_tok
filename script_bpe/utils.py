import array
import functools
import gc
import logging
import multiprocessing
import os
import sys
import time
from script_bpe.encoding.script_util import unicode_script_map
from typing import Iterable

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


# --- string/utf8 utils ---

UNASSIGNED_CATEGORIES = {"Cn", "Co", "Cs"}  # we ignore Cn=Not Assigned, Co=Private Use, Cs=Surrogate


def remove_unassigned_private_surrogate(s):
    return "".join(c for c in s if not is_unassigned_private_surrogate(c))

@functools.cache
def is_unassigned_private_surrogate(char):
    return char not in unicode_script_map()

@functools.cache
def utf_byte_type(b: int) -> int:
    start_byte = f"{b:08b}"  # cached so we can be really explicit
    if start_byte.startswith("10"):  # continuation byte
        return 0
    if start_byte.startswith("0"):
        return 1
    if start_byte.startswith("110"):
        return 2
    if start_byte.startswith("1110"):
        return 3
    if start_byte.startswith("11110"):
        return 4
    return 5  # not part of utf8
