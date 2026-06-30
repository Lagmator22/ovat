# ovat/cli/diagnostics.py
"""The real checks behind `ovat doctor`.

I keep the logic here, separate from the rendering, for one reason: every check
below actually does something (imports a module, looks on PATH, opens a socket,
validates a config), and I want to unit test that without printing tables. The
CLI command turns this list into a coloured report; this file decides pass/fail.

A check never raises. It catches its own trouble and turns it into a clear
status, because the whole point of doctor is to survive a broken setup and tell
the user what is wrong.
"""
import importlib
import os
import shutil
import socket
import sys
from dataclasses import dataclass

from ovat.config.workflow import load_workflow

# Status values a check can report. ok = good, warn = works but worth knowing,
# fail = something the user must fix before that feature works.
OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    """One diagnostic result: what I looked at, how it went, and the detail."""

    name: str
    status: str
    detail: str


# The Python packages the core toolkit imports. If any is missing, the matching
# feature cannot run, so a missing core dep is a failure, not a warning.
CORE_DEPS = ["openvino", "openvino_genai", "openai", "typer",
             "pydantic", "yaml", "fastmcp", "sqlite_vec", "rich"]


def check_python() -> Check:
    v = sys.version_info
    pretty = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 10):
        return Check("Python", OK, f"{pretty} (3.10+ required)")
    return Check("Python", FAIL, f"{pretty} is too old; OVAT needs 3.10+")


def check_core_deps() -> Check:
    missing = [name for name in CORE_DEPS if not _can_import(name)]
    if not missing:
        return Check("Core dependencies", OK, f"all {len(CORE_DEPS)} importable")
    return Check("Core dependencies", FAIL, f"missing: {', '.join(missing)}")


def check_langchain() -> Check:
    if _can_import("langchain") and _can_import("langgraph"):
        return Check("LangChain (react)", OK, "installed; agent.type: react ready")
    return Check("LangChain (react)", WARN,
                 "not installed; only needed for agent.type: react "
                 "(pip install 'ovat[langchain]')")


def check_devices() -> Check:
    """Ask OpenVINO what hardware it can see. This is the AI PC routing story."""
    try:
        import openvino as ov
        devices = ov.Core().get_available_devices()
    except Exception as exc:
        return Check("OpenVINO devices", FAIL, f"could not query devices: {exc}")
    return Check("OpenVINO devices", OK, ", ".join(devices) or "none reported")


def check_ovms_binary() -> Check:
    path = shutil.which("ovms")
    if path:
        return Check("OVMS binary", OK, path)
    return Check("OVMS binary", WARN,
                 "not on PATH; needed for 'ovat serve'/'ovat models' (AI PC only)")


def check_config(config_path: str) -> list[Check]:
    """Validate a workflow file and report what it asks for.

    Returns several checks: the config itself, then config-derived ones (the
    embeddings model on disk, and whether OVMS looks reachable). I only add the
    derived checks when the config actually opts into those features.
    """
    try:
        cfg = load_workflow(config_path)
    except FileNotFoundError:
        return [Check("Workflow config", FAIL, f"no such file: {config_path}")]
    except Exception as exc:
        # A pydantic validation error or bad YAML lands here with a clear reason.
        return [Check("Workflow config", FAIL, f"invalid: {exc}")]

    checks = [Check(
        "Workflow config", OK,
        f"model={cfg.model.name}  agent={cfg.agent.type}  "
        f"tools={[t.name for t in cfg.tools]}",
    )]

    # Only meaningful when the workflow configures RAG with the local embedder.
    if cfg.rag is not None and cfg.rag.embeddings.provider == "genai":
        model_path = cfg.rag.embeddings.model
        if os.path.exists(model_path):
            checks.append(Check("Embeddings model", OK, model_path))
        else:
            checks.append(Check("Embeddings model", WARN,
                                f"not found at {model_path}; run the export, "
                                f"then 'ovat index'"))

    checks.append(_check_ovms_reachable(cfg.model.ovms_url))
    return checks


def _check_ovms_reachable(ovms_url: str) -> Check:
    """A quick TCP connect to the OVMS host:port. No HTTP, just is-it-listening."""
    try:
        host_port = ovms_url.split("//", 1)[1].split("/", 1)[0]
        host, _, port = host_port.partition(":")
        with socket.create_connection((host, int(port or "80")), timeout=1):
            return Check("OVMS reachable", OK, f"listening at {ovms_url}")
    except OSError:
        return Check("OVMS reachable", WARN,
                     f"nothing answering at {ovms_url}; start it with 'ovat serve'")
    except (IndexError, ValueError):
        return Check("OVMS reachable", WARN, f"could not parse ovms_url: {ovms_url}")


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def run_checks(config_path: str | None = None) -> list[Check]:
    """Run every diagnostic and return the flat list of results."""
    checks = [
        check_python(),
        check_core_deps(),
        check_langchain(),
        check_devices(),
        check_ovms_binary(),
    ]
    if config_path:
        checks.extend(check_config(config_path))
    return checks
