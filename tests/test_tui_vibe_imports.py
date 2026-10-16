# tests/test_tui_vibe_imports.py
"""Every module under ovat/tui/vibe/ must import, on every Python we support.

The copied Mistral Vibe code is written for Python 3.12. OVAT runs on 3.10+,
and a `type X = ...` statement or a `datetime.UTC` import only fails when the
file is actually imported (the first is even a SyntaxError). CI's 3.10 job
once broke on exactly that, because the check only parsed the files on a newer
Python. So this test really imports each module, in a child process (the copy
touches signals, stderr and logging at import time, which a test session
should not inherit), and lists every one that fails.

EXPECTED_UNIMPORTABLE is the list of modules that cannot import yet, with the
reason. It is not a place to hide a regression: a module that is NOT listed
must import, and a listed module that now imports makes the test fail too, so
the entry is removed once the backend exists.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytest.importorskip("textual")

# module -> why it cannot import yet. Empty today: every copied module imports.
# Phase 2 adds an entry only for a module that needs a backend OVAT has not
# written yet, never to hide a Python 3.10 problem.
EXPECTED_UNIMPORTABLE: dict[str, str] = {}

# Vibe modules that were left out on purpose (they import a package OVAT will
# not depend on at module level). The copied factories that used to import them
# now do it inside the function, so everything else still imports.
NOT_COPIED: dict[str, str] = {
    "ovat.tui.vibe.cli.transcribe.mistral_transcribe_client":
        "needs the mistralai package; OVAT has its own speech to text, or none",
    "ovat.tui.vibe.cli.tts.mistral_tts_client":
        "needs the mistralai package; OVAT has its own speech output, or none",
}

_CHILD = r"""
import importlib, json, sys, traceback
failed = {}
for name in json.loads(sys.argv[1]):
    try:
        importlib.import_module(name)
    except BaseException as exc:                      # a SystemExit counts too
        failed[name] = f"{type(exc).__name__}: {exc}"
print("RESULT" + json.dumps(failed))
"""


def _modules() -> list[str]:
    """Every .py under ovat/tui/vibe, found on disk: a folder without an
    __init__.py (Vibe has a few) is a namespace package that pkgutil skips."""
    base = ROOT / "ovat"
    names = []
    for path in (base / "tui" / "vibe").rglob("*.py"):
        parts = path.relative_to(base.parent).with_suffix("").parts
        names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return sorted(names)


def _import_all(names: list[str], tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ, OVAT_TUI_HOME=str(tmp_path), PYTHONPATH=os.pathsep.join(
        [str(ROOT), os.environ.get("PYTHONPATH", "")]))
    proc = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(names)],
                          capture_output=True, text=True, env=env, timeout=300)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    assert lines, f"the import child died:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    return json.loads(lines[-1][len("RESULT"):])


def test_the_copied_tree_is_not_empty():
    assert len(_modules()) > 100


def test_every_copied_module_imports_on_this_python(tmp_path):
    names = _modules()
    failed = _import_all(names, tmp_path)

    unexpected = {n: why for n, why in failed.items() if n not in EXPECTED_UNIMPORTABLE}
    assert not unexpected, (
        f"{len(unexpected)} module(s) cannot be imported on "
        f"Python {sys.version_info.major}.{sys.version_info.minor}:\n"
        + "\n".join(f"  {n}: {why}" for n, why in sorted(unexpected.items())))

    stale = [n for n in EXPECTED_UNIMPORTABLE if n not in failed]
    assert not stale, f"these now import, drop them from EXPECTED_UNIMPORTABLE: {stale}"


def test_modules_left_out_on_purpose_stay_out():
    names = set(_modules())
    back = [n for n in NOT_COPIED if n in names]
    assert not back, (
        f"{back} came back: they import a package OVAT does not depend on, so "
        "port them (guard the import) and drop them from NOT_COPIED")


def test_chatscroll_is_still_importable_from_the_full_app():
    from ovat.tui.vibe.cli.textual_ui.app import ChatScroll

    assert ChatScroll.__name__ == "ChatScroll"
