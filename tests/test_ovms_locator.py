# tests/test_ovms_locator.py
"""Tests for the ovms binary resolution: config → env → PATH → known spots.

Note to myself: the whole point is that a Windows setupvars.bat install
(ovms unzipped in some folder, never on PATH) still just works. I fake a
binary with tmp files; nothing here runs ovms.
"""
import os

from ovat.core import ovms_locator
from ovat.core.ovms_locator import find_ovms


def _fake_binary(tmp_path, name=None):
    exe = tmp_path / (name or ovms_locator._EXE)
    exe.write_text("#!/bin/sh\n")
    return str(exe)


def test_explicit_file_path_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("OVAT_OVMS", "/somewhere/else")   # must be ignored
    exe = _fake_binary(tmp_path)
    path, how = find_ovms(exe)
    assert path == exe
    assert "config" in how


def test_explicit_folder_is_accepted(tmp_path):
    exe = _fake_binary(tmp_path)
    path, how = find_ovms(str(tmp_path))                 # folder, not file
    assert path == exe


def test_explicit_but_wrong_path_reports_it(tmp_path):
    path, how = find_ovms(str(tmp_path / "nope"))
    assert path is None
    assert "ovms_binary" in how                          # names the culprit


def test_env_var_is_used_when_no_explicit(tmp_path, monkeypatch):
    exe = _fake_binary(tmp_path)
    monkeypatch.setenv("OVAT_OVMS", str(tmp_path))
    path, how = find_ovms()
    assert path == exe
    assert "OVAT_OVMS" in how


def test_path_lookup_still_works(monkeypatch, tmp_path):
    monkeypatch.delenv("OVAT_OVMS", raising=False)
    exe = _fake_binary(tmp_path)
    monkeypatch.setattr(ovms_locator.shutil, "which", lambda name: exe)
    path, how = find_ovms()
    assert path == exe and how == "PATH"


def test_known_locations_are_the_last_resort(monkeypatch, tmp_path):
    monkeypatch.delenv("OVAT_OVMS", raising=False)
    monkeypatch.setattr(ovms_locator.shutil, "which", lambda name: None)
    exe = _fake_binary(tmp_path)
    monkeypatch.setattr(ovms_locator, "_KNOWN_DIRS", [str(tmp_path)])
    path, how = find_ovms()
    assert path == exe
    assert "known location" in how


def test_ovms_is_found_where_the_readme_tells_users_to_put_it(monkeypatch, tmp_path):
    """The README's own install steps must produce a findable OVMS.

    README.md says, run from inside the clone:

        curl -L ...ovms_windows_....zip -o ovms.zip
        tar -xf ovms.zip

    which lands OVMS at <repo>/ovms. The search list held only ~/ovms_windows,
    ~/ovms, C:\\ovms and C:\\ovms_windows, so a user who followed the
    instructions EXACTLY got "not found (config, OVAT_OVMS, PATH, known
    folders all empty)" and had to set OVAT_OVMS by hand.

    Caught in the AI PC clean-room run: it only passed there because that
    machine happened to already have ~/ovms_windows from an earlier manual
    install -- the textbook "works on your laptop" pass.
    """
    monkeypatch.delenv("OVAT_OVMS", raising=False)
    monkeypatch.setattr(ovms_locator.shutil, "which", lambda name: None)
    project = tmp_path / "ovat"
    (project / "ovms").mkdir(parents=True)
    exe = _fake_binary(project / "ovms")
    monkeypatch.chdir(project)
    path, how = find_ovms()
    assert path == exe, "OVMS unpacked per the README was not found"
    assert "known location" in how


def test_nothing_found_says_where_it_looked(monkeypatch):
    monkeypatch.delenv("OVAT_OVMS", raising=False)
    monkeypatch.setattr(ovms_locator.shutil, "which", lambda name: None)
    monkeypatch.setattr(ovms_locator, "_KNOWN_DIRS", [])
    path, how = find_ovms()
    assert path is None
    assert "not found" in how


def test_model_server_launches_the_resolved_binary(monkeypatch, tmp_path):
    # The binary the locator finds must be the argv[0] OVMS starts with, and
    # its folder must lead the child's PATH (the setupvars.bat replacement).
    from ovat.core.model_server import ModelServer

    captured = {}

    class FakePopen:
        pid = 1

        def __init__(self):
            pass

    def fake_popen(cmd, env=None, **kwargs):
        captured["cmd"], captured["env"] = cmd, env
        return FakePopen()

    monkeypatch.setattr("ovat.core.model_server.subprocess.Popen", fake_popen)
    exe = _fake_binary(tmp_path)
    server = ModelServer(model_name="m", binary=exe)
    server.start(log_path=str(tmp_path / "l.log"), pid_path=str(tmp_path / "p.pid"))
    assert captured["cmd"][0] == exe
    assert captured["env"]["PATH"].startswith(str(tmp_path))
