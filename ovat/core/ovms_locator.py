# ovat/core/ovms_locator.py
"""Find the ovms executable, wherever a real machine actually keeps it.

The reality on Windows AI PCs: OVMS is unzipped somewhere like
C:\\Users\\you\\ovms_windows and set up with setupvars.bat; it is almost
never on PATH. So "ovms must be on PATH" fails for basically every new user.

Resolution order (first hit wins), returning (path, how) so doctor and error
messages can say exactly where it came from:
  1. an explicit path (the `model.ovms_binary` YAML field): file or folder
  2. the OVAT_OVMS environment variable: same semantics
  3. PATH (shutil.which): the classic case
  4. well-known install folders per OS

Launching by full path works without setupvars.bat for the common case:
Windows searches an exe's own directory for its DLLs first, and start()
additionally prepends the binary's folder to the child's PATH.
"""
import os
import shutil
import sys

# Where people actually unzip OVMS. Checked LAST, after explicit/env/PATH.
_EXE = "ovms.exe" if sys.platform == "win32" else "ovms"
_KNOWN_DIRS = [
    os.path.expanduser(p) for p in (
        [r"~\ovms_windows", r"~\ovms", r"C:\ovms", r"C:\ovms_windows"]
        if sys.platform == "win32" else
        ["~/ovms/bin", "~/ovms", "/opt/ovms/bin", "/usr/local/bin"]
    )
]


def _as_binary(path: str) -> str | None:
    """Accept a file OR a folder; return the executable path if it exists."""
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return path
    candidate = os.path.join(path, _EXE)
    if os.path.isfile(candidate):
        return candidate
    return None


def find_ovms(explicit: str | None = None) -> tuple[str | None, str]:
    """Resolve the ovms binary. Returns (path or None, how it was found).

    `how` is a short human phrase ('config', 'OVAT_OVMS env', 'PATH',
    'known location', or a hint when nothing was found) so doctor and the
    serve/models errors can tell the user exactly what happened.
    """
    if explicit:
        found = _as_binary(explicit)
        if found:
            return found, "config (model.ovms_binary)"
        return None, (f"model.ovms_binary points at {explicit!r} but no "
                      f"{_EXE} exists there")

    env = os.environ.get("OVAT_OVMS")
    if env:
        found = _as_binary(env)
        if found:
            return found, "OVAT_OVMS env"
        return None, f"OVAT_OVMS points at {env!r} but no {_EXE} exists there"

    on_path = shutil.which("ovms")
    if on_path:
        return on_path, "PATH"

    for folder in _KNOWN_DIRS:
        found = _as_binary(folder)
        if found:
            return found, f"known location ({folder})"

    return None, "not found (config, OVAT_OVMS, PATH, known folders all empty)"
