import signal
import subprocess

import pytest

from paper_utils.boundary import run_bilingual


def test_parse_process_group_rss_sums_only_requested_group():
    output = """  101  101  1500
  102  101  2500
  103  999  8000
bad row
"""
    assert run_bilingual.parse_process_group_rss(output, 101) == 4_000 * 1024


def test_process_group_rss_uses_ps_machine_columns():
    calls = []

    def fake_ps(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="  101  101  42\n")

    assert run_bilingual.process_group_rss_bytes(101, fake_ps) == 42 * 1024
    assert calls == [(["ps", "-axo", "pid=,pgid=,rss="], {"check": True, "capture_output": True, "text": True})]


def test_terminate_process_group_escalates_after_timeout():
    class Process:
        pid = 456

        def __init__(self):
            self.wait_calls = []

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired("child", timeout)

    process = Process()
    killed = []
    assert run_bilingual.terminate_process_group(process, killpg=lambda pgid, sig: killed.append((pgid, sig)))
    assert killed == [(456, signal.SIGTERM), (456, signal.SIGKILL)]
    assert process.wait_calls == [5, 5]


def test_terminate_process_group_still_targets_children_after_leader_exits():
    class Process:
        pid = 654

        def poll(self):
            return 0

        def wait(self, timeout=None):
            assert timeout == 5

    killed = []
    assert run_bilingual.terminate_process_group(Process(), killpg=lambda pgid, sig: killed.append((pgid, sig)))
    assert killed == [(654, signal.SIGTERM), (654, signal.SIGKILL)]


def test_monitor_child_kills_group_and_reports_limit(monkeypatch):
    class Process:
        pid = 789
        args = ["child"]

        def poll(self):
            return None

    process = Process()
    killed = []
    monkeypatch.setattr(run_bilingual, "terminate_process_group", lambda value: killed.append(value.pid))
    polls = []
    with pytest.raises(run_bilingual.MemoryLimitExceeded, match="--max-rss-gib"):
        run_bilingual.monitor_child(process, 100, lambda rss, highwater: polls.append((rss, highwater)),
                                    rss_reader=lambda pgid: 101, sleep=lambda seconds: None)
    assert polls == [(101, 101)]
    assert killed == [789]
