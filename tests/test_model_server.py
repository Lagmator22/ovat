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

from ovat.core import model_server
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


def test_a_download_that_keeps_moving_is_never_timed_out(monkeypatch, tmp_path):
    """No absolute cap: a model download can legitimately take all night.

    The old 120s cap could not cover even a fast first run. Measured on the
    AI PC, 2026-08-03, serving Qwen3.5-4B from scratch: ~185s needed against
    that 120s cap, so `ovat serve` could not succeed on any machine that had
    to download -- i.e. every new machine.

    Raising the number is not the fix either. Download time is unbounded: it
    depends on the link, and on a slow one this is hours. Any fixed value is
    either too small for a slow link or so large it is useless for detecting
    a genuine hang. This is the same reason curl grew --speed-time/-limit and
    wget grew --read-timeout: watch PROGRESS, not elapsed time.

    So the clock only runs while nothing is happening. Here the server keeps
    writing to its log, so it must still be waiting after well past any fixed
    cap would have fired.
    """
    from ovat.core.model_server import ModelServer

    server = ModelServer(model_name="m")
    server.process = None
    log = tmp_path / "ovms.log"
    log.write_text("start")
    server.log_path = str(log)

    ticks = {"n": 0}
    clock = {"t": 0.0}

    def fake_sleep(seconds):
        clock["t"] += seconds
        ticks["n"] += 1
        # Still downloading: the log grows on every poll.
        log.write_text("x" * (ticks["n"] * 1000))

    monkeypatch.setattr("ovat.core.model_server.time.sleep", fake_sleep)
    monkeypatch.setattr("ovat.core.model_server.time.time", lambda: clock["t"])

    # A 60s stall budget, but progress every poll, so it must run far past it.
    server.wait_until_ready(stall_timeout=60, max_polls=200)
    # The real assertion is about TIME, not poll count: it kept waiting for
    # simulated minutes, well beyond both the 60s stall budget and the old
    # 120s hard cap, purely because the download kept moving.
    assert clock["t"] > 300, (
        f"gave up after {clock['t']}s despite continuous progress; a stall "
        f"budget must not behave like a deadline")


def test_a_genuinely_stalled_server_still_gives_up(monkeypatch, tmp_path):
    """The other half: no progress at all must not wait forever."""
    from ovat.core.model_server import ModelServer

    server = ModelServer(model_name="m")
    server.process = None
    log = tmp_path / "ovms.log"
    log.write_text("start")          # never grows again
    server.log_path = str(log)

    clock = {"t": 0.0}
    monkeypatch.setattr("ovat.core.model_server.time.sleep",
                        lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr("ovat.core.model_server.time.time", lambda: clock["t"])

    assert server.wait_until_ready(stall_timeout=60) is False
    assert clock["t"] < 200, "waited far longer than the stall budget"


def test_readiness_reports_progress_while_it_waits(monkeypatch, tmp_path):
    """Silence then a failure reads as a hang, not as a download."""
    from ovat.core.model_server import ModelServer

    server = ModelServer(model_name="m")
    server.process = None
    server.log_path = str(tmp_path / "missing.log")
    seen = []
    clock = {"t": 0.0}
    monkeypatch.setattr("ovat.core.model_server.time.sleep",
                        lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr("ovat.core.model_server.time.time", lambda: clock["t"])
    server.wait_until_ready(stall_timeout=10,
                            on_progress=lambda elapsed: seen.append(elapsed))
    assert seen, "wait_until_ready never reported progress"


def test_tool_parser_auto_lets_ovms_choose(monkeypatch, tmp_path):
    """`auto` must OMIT --tool_parser, not pass the string "auto".

    OVMS reads the model's chat template at startup and picks the right
    parser itself, but only when the flag is absent -- an explicit value
    always wins. OVAT passed --tool_parser unconditionally, so a config that
    still said "hermes3" would override a correct auto-detection with a wrong
    parser and tool calls would silently stop being decoded.

    This is not hypothetical. Qwen3.5 emits

        <tool_call><function=name><parameter=k>v</parameter></function></tool_call>

    which is the qwen3coder shape, NOT hermes3's JSON body. OVMS ships no
    qwen3_5 parser at all, so letting the server decide is the only forward
    compatible answer for a family newer than OVAT's own parser list.
    """
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakePopen()

    monkeypatch.setattr("ovat.core.model_server.subprocess.Popen", fake_popen)

    auto = ModelServer(model_name="m", tool_parser="auto")
    auto.start(log_path=str(tmp_path / "a.log"), pid_path=str(tmp_path / "a.pid"))
    assert "--tool_parser" not in captured["cmd"]
    assert "auto" not in captured["cmd"]           # never passed as a value

    explicit = ModelServer(model_name="m", tool_parser="hermes3")
    explicit.start(log_path=str(tmp_path / "b.log"), pid_path=str(tmp_path / "b.pid"))
    assert "--tool_parser" in captured["cmd"]      # an override still works
    assert "hermes3" in captured["cmd"]


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


def test_liveness_never_sends_signal_zero(tmp_path, monkeypatch):
    """`os.kill(pid, 0)` is the classic POSIX existence check that sends
    nothing. On Windows it is a console grenade.

    Windows has no signals, so CPython maps os.kill(pid, sig) onto
    GenerateConsoleCtrlEvent whenever sig is 0 or 1, because
    signal.CTRL_C_EVENT == 0 and CTRL_BREAK_EVENT == 1. The call therefore
    delivers Ctrl-C to EVERY process attached to the console, not to `pid`:
    the caller, the shell it was typed into, and any test runner above it.

    Measured on the AI PC: one os.kill(pid, 0) killed both the probe process
    and its child with KeyboardInterrupt, and because the stale-check below
    polls every 0.2s for wait_seconds, `ovat serve --stop` fired about fifty
    of them. It is why a full `pytest -q` died with exit 137 the moment it
    reached this file, and it took the developer's shell with it.

    os.kill is stubbed here so the test is safe to RUN while red: it records
    the signals the code would have sent instead of sending them.

    The stub records the CALLER, and only OVAT's own frames are asserted on.
    psutil.pid_exists() legitimately uses os.kill(pid, 0) on POSIX, where it
    really is the harmless existence check; the danger is Windows-only, and
    psutil uses a different mechanism there. Patching os.kill catches psutil
    too, so without the caller filter this test failed on macOS and Linux
    while passing on the one platform the bug actually affects, which is the
    worst possible way round.
    """
    import sys

    calls = []

    def recording_kill(pid, sig):
        calls.append((pid, sig, sys._getframe(1).f_code.co_filename))

    monkeypatch.setattr(model_server.os, "kill", recording_kill)

    pid_path = tmp_path / "ovms.pid"
    # Our own pid: certainly alive, so the liveness branch is really taken.
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    stop_from_pidfile(str(pid_path), wait_seconds=0)

    signals_sent = [sig for _pid, sig, caller in calls
                    if caller.endswith("model_server.py")]
    assert 0 not in signals_sent, (
        "liveness was probed with signal 0, which is CTRL_C_EVENT on Windows "
        "and Ctrl-Cs the whole console")
    # It must still have done its job, or the assertion above is vacuous.
    assert signals_sent, "no signal reached the process at all"
    assert 1 not in signals_sent, "signal 1 is CTRL_BREAK_EVENT on Windows"


def test_a_live_pid_reads_as_running_and_a_dead_one_does_not():
    """The replacement probe must still answer the question correctly."""
    assert model_server._pid_is_running(os.getpid()) is True

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    assert model_server._pid_is_running(dead.pid) is False


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


def test_ovat_never_passes_signal_zero_or_one_anywhere():
    """A source-level backstop for the whole module.

    The behavioural test above only covers stop_from_pidfile. Signals 0 and 1
    are CTRL_C_EVENT and CTRL_BREAK_EVENT on Windows, so ANY os.kill with
    either value is a console grenade wherever it appears, and the next one
    would be just as invisible from a POSIX machine as the last one was.
    """
    import re
    from pathlib import Path

    source = Path(model_server.__file__).read_text(encoding="utf-8")
    # Strip comments and docstrings: this file EXPLAINS the bug at length.
    code = re.sub(r"#.*", "", source)
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    for match in re.finditer(r"os\.kill\([^,]+,\s*([^)]+)\)", code):
        argument = match.group(1).strip()
        assert argument not in {"0", "1"}, (
            f"os.kill(..., {argument}) sends a console Ctrl-C on Windows")
