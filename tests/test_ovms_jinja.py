# tests/test_ovms_jinja.py
"""OVMS on Linux needs Jinja2 in the SYSTEM Python, and nothing said so.

The python_on build embeds the system python3 (OVMS's own bare-metal guide,
v2026.4.1, tells every Linux user to `pip3 install "Jinja2==3.1.6"
"MarkupSafe==3.0.2"`). The Linux archives ship neither. Without them OVMS
loads, logs "Chat template has not been loaded properly", reports READY, and
then answers every chat request with a 400. These tests cover the three places
the user is now told: `ovat doctor`, `ovat serve` and `ovat setup`.

No network and no real OVMS: the Python probe and the log are faked.
"""
import os
import subprocess

from typer.testing import CliRunner

from ovat.cli import diagnostics
from ovat.cli.main import app
from ovat.core import model_server

runner = CliRunner()

PIP = 'pip3 install "Jinja2==3.1.6" "MarkupSafe==3.0.2"'
LOG_LINE = ("[2026-10-16 10:00:00.000][1][serving][warning]Warning: Chat "
            "template has not been loaded properly. Servable will not "
            "respond to /chat/completions endpoint.\n")


def _fake_install(tmp_path, libpython=b"libpython3.12.so.1.0", python_on=True):
    """<root>/bin/ovms (+ <root>/lib/python for a python_on layout)."""
    root = tmp_path / "ovms"
    (root / "bin").mkdir(parents=True)
    if python_on:
        (root / "lib" / "python").mkdir(parents=True)
    binary = root / "bin" / "ovms"
    binary.write_bytes(b"\x7fELF" + b"\0" * 64 + libpython + b"\0jinja2\0")
    return str(binary)


def _probe(monkeypatch, returncode=0, stderr="", seen=None):
    """Fake the system-Python probe; record what it was asked to run."""
    monkeypatch.setattr(model_server.sys, "platform", "linux")
    monkeypatch.setattr(model_server, "_find_system_python",
                        lambda name: f"/usr/bin/{name}")

    def run(cmd, **kwargs):
        if seen is not None:
            seen.update(cmd=cmd, env=kwargs.get("env"))
        return subprocess.CompletedProcess(cmd, returncode, "", stderr)

    monkeypatch.setattr(model_server.subprocess, "run", run)


NO_JINJA = "ModuleNotFoundError: No module named 'jinja2'"


# --- the probe itself ------------------------------------------------------

def test_it_reads_which_libpython_the_binary_links(tmp_path):
    assert model_server._linked_python_minor(_fake_install(tmp_path)) == "3.12"
    other = _fake_install(tmp_path / "b", libpython=b"libpython3.10.so.1.0")
    assert model_server._linked_python_minor(other) == "3.10"


def test_a_missing_jinja_is_reported_for_the_python_the_binary_links(
        tmp_path, monkeypatch):
    seen = {}
    _probe(monkeypatch, returncode=1, stderr=NO_JINJA, seen=seen)
    state, python = model_server.jinja_status(_fake_install(tmp_path))
    assert (state, python) == ("missing", "python3.12")
    assert seen["cmd"][0] == "/usr/bin/python3.12"
    assert "jinja2" in seen["cmd"][-1] and "markupsafe" in seen["cmd"][-1]


def test_an_importable_jinja_is_ok(tmp_path, monkeypatch):
    _probe(monkeypatch, returncode=0)
    assert model_server.jinja_status(_fake_install(tmp_path)) == (
        "ok", "python3.12")


def test_the_probe_runs_under_the_env_ovms_gets_so_a_pythonpath_folder_counts(
        tmp_path, monkeypatch):
    """Installing Jinja2 into a folder and exporting PYTHONPATH made a broken
    server answer, so the probe must see the same PYTHONPATH OVMS will."""
    seen = {}
    _probe(monkeypatch, returncode=0, seen=seen)
    monkeypatch.setenv("PYTHONPATH", "/home/u/pylibs")
    model_server.jinja_status(_fake_install(tmp_path))
    assert "/home/u/pylibs" in seen["env"]["PYTHONPATH"].split(os.pathsep)


def test_a_python_that_cannot_be_run_is_unknown_not_missing(tmp_path,
                                                            monkeypatch):
    _probe(monkeypatch, returncode=1, stderr="Segmentation fault")
    assert model_server.jinja_status(_fake_install(tmp_path))[0] == "unknown"
    monkeypatch.setattr(model_server, "_find_system_python", lambda name: None)
    assert model_server.jinja_status(_fake_install(tmp_path / "b"))[0] == (
        "unknown")


def test_only_a_linux_python_on_install_is_checked(tmp_path, monkeypatch):
    _probe(monkeypatch, returncode=1, stderr=NO_JINJA)
    # No lib/python: not the python_on layout (not verified for python_off).
    off = _fake_install(tmp_path, python_on=False)
    assert model_server.jinja_status(off)[0] == "n/a"
    # Windows ships its own python folder with jinja2 inside (archive listing).
    monkeypatch.setattr(model_server.sys, "platform", "win32")
    assert model_server.jinja_status(_fake_install(tmp_path / "b"))[0] == "n/a"


def test_the_log_line_is_found_and_its_absence_is_not_a_warning(tmp_path):
    log = tmp_path / "ovms.log"
    log.write_text("starting\n" + LOG_LINE, encoding="utf-8")
    assert model_server.chat_template_failed(str(log)) is True
    log.write_text("starting\nall good\n", encoding="utf-8")
    assert model_server.chat_template_failed(str(log)) is False
    assert model_server.chat_template_failed(str(tmp_path / "none")) is False


# --- ovat doctor -----------------------------------------------------------

def _doctor(monkeypatch, tmp_path, state):
    monkeypatch.setattr(diagnostics.sys, "platform", "linux")
    import ovat.core.ovms_locator as locator
    binary = _fake_install(tmp_path)
    monkeypatch.setattr(locator, "find_ovms", lambda e=None: (binary, "test"))
    monkeypatch.setattr(model_server, "jinja_status",
                        lambda b: (state, "python3.12"))
    return diagnostics.check_ovms_python()


def test_doctor_warns_when_jinja_is_missing_and_gives_the_docs_command(
        monkeypatch, tmp_path):
    check = _doctor(monkeypatch, tmp_path, "missing")
    assert check.status == diagnostics.WARN
    assert PIP in check.detail
    assert "python3.12" in check.detail


def test_doctor_is_ok_when_jinja_is_importable(monkeypatch, tmp_path):
    check = _doctor(monkeypatch, tmp_path, "ok")
    assert check.status == diagnostics.OK


def test_doctor_says_not_verified_when_it_could_not_look(monkeypatch, tmp_path):
    check = _doctor(monkeypatch, tmp_path, "unknown")
    assert check.status == diagnostics.WARN
    assert "not verified" in check.detail


def test_doctor_stays_quiet_where_the_check_does_not_apply(monkeypatch,
                                                           tmp_path):
    assert _doctor(monkeypatch, tmp_path, "n/a") is None
    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    assert diagnostics.check_ovms_python() is None


def test_the_doctor_report_includes_the_row(monkeypatch, tmp_path):
    _doctor(monkeypatch, tmp_path, "missing")
    names = {c.name: c for c in diagnostics.run_checks()}
    assert names["OVMS Python (Jinja2)"].status == diagnostics.WARN


# --- ovat serve ------------------------------------------------------------

def _serve(monkeypatch, tmp_path, log_text, state="missing", platform="linux"):
    from ovat.cli import main as cli_main
    from ovat.core import model_scout, ovms_locator
    monkeypatch.setattr(cli_main.sys, "platform", platform)
    monkeypatch.setattr(model_server, "jinja_status",
                        lambda b: (state, "python3.12"))
    log = tmp_path / "ovms.log"
    log.write_text(log_text, encoding="utf-8")
    cfg = tmp_path / "w.yml"
    cfg.write_text("model:\n  name: X\nagent:\n  type: native\n",
                   encoding="utf-8")

    class FakeModelServer:
        DEFAULT_STALL_TIMEOUT = 120

        def __init__(self, **kwargs):
            self.port = 8000
            self.base_url = "http://localhost:8000/v3"
            self.log_path = str(log)

            class Proc:
                pid = 1234

                def poll(self):
                    return None
            self.process = Proc()

        def already_serving(self):
            return None

        def start(self, *args, **kwargs):
            pass

        def wait_until_ready(self, *args, **kwargs):
            return True

        def stop(self):
            pass

    monkeypatch.setattr(model_server, "ModelServer", FakeModelServer)
    monkeypatch.setattr(model_scout, "identify_model", lambda *a: "llm")
    monkeypatch.setattr(ovms_locator, "find_ovms",
                        lambda *a: ("/path/to/ovms", "test"))
    return runner.invoke(app, ["serve", str(cfg)])


def test_serve_warns_when_ovms_says_it_cannot_load_the_chat_template(
        monkeypatch, tmp_path):
    result = _serve(monkeypatch, tmp_path, "loading\n" + LOG_LINE)
    flat = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert "OVMS is ready" in flat                   # it still is ready
    assert "could not load the chat template" in flat
    assert "400" in flat
    assert PIP in flat                               # the fix, spelled out
    assert "PYTHONPATH" in flat                      # and the way around pip


def test_serve_is_quiet_when_the_log_has_no_such_line(monkeypatch, tmp_path):
    result = _serve(monkeypatch, tmp_path, "loading\nready\n")
    flat = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert "OVMS is ready" in flat
    assert "chat template" not in flat.lower()
    assert "pip3" not in flat


def test_serve_off_linux_points_at_the_log_and_does_not_blame_jinja(
        monkeypatch, tmp_path):
    result = _serve(monkeypatch, tmp_path, LOG_LINE, state="n/a",
                    platform="win32")
    flat = " ".join(result.output.split())
    assert "could not load the chat template" in flat
    assert "pip3" not in flat


# --- ovat setup ------------------------------------------------------------

def _setup(monkeypatch, tmp_path, state):
    from ovat.core import ovms_installer
    monkeypatch.setattr(ovms_installer, "installed_binary", lambda root=None: None)
    monkeypatch.setattr(ovms_installer, "install",
                        lambda **kw: ("/somewhere/ovms", "installed"))
    monkeypatch.setattr(ovms_installer, "asset_for_platform",
                        lambda version=None: ("ovms_test.tar.gz", "file:///x"))
    monkeypatch.setattr(ovms_installer, "linux_support_note", lambda: None)
    monkeypatch.setattr(model_server, "jinja_status",
                        lambda b: (state, "python3.12"))
    return runner.invoke(app, ["setup", "--yes"])


def test_setup_prints_the_jinja_step_when_the_system_python_lacks_it(
        monkeypatch, tmp_path):
    result = _setup(monkeypatch, tmp_path, "missing")
    flat = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert PIP in flat
    assert "python3.12" in flat


def test_setup_says_nothing_extra_when_jinja_is_there(monkeypatch, tmp_path):
    result = _setup(monkeypatch, tmp_path, "ok")
    assert result.exit_code == 0, result.output
    assert "pip3" not in result.output
