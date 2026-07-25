# tests/test_model_server.py
"""Tests for the OVMS process lifecycle: start's pidfile, stop's escalation,
and the pidfile-based stop that `ovat serve --stop` uses.

Note to myself: I never need a real `ovms` binary here. start() is tested by
monkeypatching subprocess.Popen with a stand-in; stop_from_pidfile() is tested
against a REAL tiny `python -c sleep` process, because the interesting part
(os.kill on a pid from a file) should run for real, not against a mock.
"""
import os
import subprocess
import sys

import pytest

from ovat.core.model_server import ModelServer, stop_from_pidfile


class FakePopen:
    """A stand-in process: records what stop() did to it."""

    def __init__(self, pid=4242, stubborn=False):
        self.pid = pid
        self.stubborn = stubborn        # ignores terminate() like a hung OVMS
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        # A stubborn process survives terminate(); only kill() ends it.
        if self.stubborn and not self.killed:
            raise subprocess.TimeoutExpired(cmd="ovms", timeout=timeout)
        return 0


def _server():
    return ModelServer(model_name="m", model_repository_path="models")


def _start_with_fake(monkeypatch, tmp_path, fake):
    # Patch Popen where model_server looks it up, so start() "launches" our fake.
    monkeypatch.setattr("ovat.core.model_server.subprocess.Popen",
                        lambda *a, **k: fake)
    server = _server()
    log = str(tmp_path / "ovms.log")
    pid = str(tmp_path / "ovms.pid")
    server.start(log_path=log, pid_path=pid)
    return server, log, pid


def test_start_writes_the_pidfile(monkeypatch, tmp_path):
    server, _, pid_path = _start_with_fake(monkeypatch, tmp_path, FakePopen(pid=777))
    assert open(pid_path, encoding="utf-8").read() == "777"


def test_prefix_caching_is_a_knob_not_a_constant(monkeypatch, tmp_path):
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakePopen()

    monkeypatch.setattr("ovat.core.model_server.subprocess.Popen", fake_popen)

    on = ModelServer(model_name="m")                      # default: enabled
    on.start(log_path=str(tmp_path / "a.log"), pid_path=str(tmp_path / "a.pid"))
    assert "--enable_prefix_caching" in captured["cmd"]

    off = ModelServer(model_name="m", enable_prefix_caching=False)
    off.start(log_path=str(tmp_path / "b.log"), pid_path=str(tmp_path / "b.pid"))
    assert "--enable_prefix_caching" not in captured["cmd"]


def test_creation_flags_detach_ovms_on_windows_only(monkeypatch, tmp_path):
    """serve must leave OVMS running after the console that started it goes.

    On Windows a child is attached to the parent's console, so closing the
    window or Ctrl-C would take OVMS with it. Off Windows the value must stay
    0: subprocess raises ValueError on POSIX for any non-zero creationflags,
    so getting this wrong breaks `ovat serve` on Linux, not just Windows.
    """
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured.update(kwargs)
        return FakePopen()

    monkeypatch.setattr("ovat.core.model_server.subprocess.Popen", fake_popen)

    monkeypatch.setattr("ovat.core.model_server.os.name", "posix")
    _server().start(log_path=str(tmp_path / "a.log"),
                    pid_path=str(tmp_path / "a.pid"))
    assert captured["creationflags"] == 0

    # The real Win32 values; they do not exist as attributes off Windows,
    # which is why the lookup has to stay inside the os.name guard.
    monkeypatch.setattr("ovat.core.model_server.os.name", "nt")
    monkeypatch.setattr("ovat.core.model_server.subprocess.DETACHED_PROCESS",
                        0x00000008, raising=False)
    monkeypatch.setattr(
        "ovat.core.model_server.subprocess.CREATE_NEW_PROCESS_GROUP",
        0x00000200, raising=False)
    _server().start(log_path=str(tmp_path / "b.log"),
                    pid_path=str(tmp_path / "b.pid"))
    assert captured["creationflags"] == 0x00000008 | 0x00000200


def test_stop_polite_path_closes_log_and_removes_pidfile(monkeypatch, tmp_path):
    fake = FakePopen()
    server, log_path, pid_path = _start_with_fake(monkeypatch, tmp_path, fake)
    server.stop()
    assert fake.terminated and not fake.killed     # SIGTERM was enough
    assert server._log_file is None                # handle released
    assert not os.path.exists(pid_path)            # no stale pidfile left


def test_stop_escalates_to_kill_and_still_cleans_up(monkeypatch, tmp_path):
    fake = FakePopen(stubborn=True)
    server, log_path, pid_path = _start_with_fake(monkeypatch, tmp_path, fake)
    server.stop()
    assert fake.terminated and fake.killed         # asked, then forced
    # The old bug: TimeoutExpired skipped the close and leaked the handle.
    assert server._log_file is None
    assert not os.path.exists(pid_path)


# stop_from_pidfile: the `ovat serve --stop` path, against a real process

def test_stop_from_pidfile_stops_a_real_process(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid_path = str(tmp_path / "ovms.pid")
    with open(pid_path, "w", encoding="utf-8") as f:
        f.write(str(proc.pid))

    # Zombie gotcha: this sleeper is OUR child, and a dead-but-unreaped child
    # still answers os.kill(pid, 0), so the liveness poll would say "alive"
    # forever. A real `ovat serve --stop` stops an orphan that init reaps
    # instantly. This thread plays init's role and reaps the moment it dies.
    import threading
    threading.Thread(target=proc.wait, daemon=True).start()

    message = stop_from_pidfile(pid_path)

    assert "Stopped" in message
    assert proc.wait(timeout=5) is not None        # it is really gone
    assert not os.path.exists(pid_path)


def test_stop_from_pidfile_missing_file_is_a_calm_message(tmp_path):
    message = stop_from_pidfile(str(tmp_path / "nope.pid"))
    assert "nothing to stop" in message


def test_stop_from_pidfile_stale_pid_removes_the_file(tmp_path):
    # A pid that is certainly dead: spawn something and let it finish first.
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    pid_path = str(tmp_path / "ovms.pid")
    with open(pid_path, "w", encoding="utf-8") as f:
        f.write(str(proc.pid))

    message = stop_from_pidfile(pid_path)

    assert "not running" in message
    assert not os.path.exists(pid_path)


def test_stop_from_pidfile_corrupt_file_is_removed(tmp_path):
    pid_path = str(tmp_path / "ovms.pid")
    with open(pid_path, "w", encoding="utf-8") as f:
        f.write("not-a-number")
    assert "corrupt" in stop_from_pidfile(pid_path)
    assert not os.path.exists(pid_path)
