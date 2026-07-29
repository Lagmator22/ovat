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


# One row per framework engine: (label, agent.type, extra, modules needed).
#
# This was a single hand-written LangChain check, which was the whole story
# while react was the only framework engine. Since W7-W8 there are four, and a
# box missing LlamaIndex or the OpenAI Agents SDK still got "All clear" from
# doctor and then two failed rows from `ovat bench`. doctor exists so the user
# learns what is ready HERE before something else fails to tell them.
#
# The engines themselves live in factory.AGENT_TYPES; this table adds only the
# per-engine install advice, which is presentation, and check_engines() below
# asserts the two stay in step.
FRAMEWORK_ENGINES = (
    ("LangChain (react)", "react", "langchain", ("langchain", "langgraph")),
    ("LlamaIndex", "llamaindex", "llamaindex",
     ("llama_index.core", "llama_index.llms.openai_like")),
    ("OpenAI Agents SDK", "openai-agents", "openai-agents", ("agents",)),
)


def check_engines() -> list[Check]:
    """One row per framework engine, saying whether it can run here."""
    checks = []
    for label, agent_type, extra, modules in FRAMEWORK_ENGINES:
        if all(_can_import(m) for m in modules):
            checks.append(Check(label, OK,
                                f"installed; agent.type: {agent_type} ready"))
        else:
            checks.append(Check(label, WARN,
                                f"not installed; only needed for agent.type: "
                                f"{agent_type} (pip install 'ovat[{extra}]')"))
    return checks


def check_devices() -> Check:
    """Ask OpenVINO what hardware it can see. This is the AI PC routing story."""
    try:
        import openvino as ov
        devices = ov.Core().get_available_devices()
    except Exception as exc:
        return Check("OpenVINO devices", FAIL, f"could not query devices: {exc}")
    return Check("OpenVINO devices", OK, ", ".join(devices) or "none reported")


def check_device_routing() -> Check:
    """Where each model type WOULD run here (the Layer 9 routing table).

    This is DeviceManager doing its job in front of the user: LLM prefers
    GPU (tool calling, KV cache), embeddings prefer NPU (static shapes, low
    power), whisper stays on CPU, and everything falls back to CPU.
    """
    try:
        from ovat.core.device_manager import DeviceManager
        summary = DeviceManager().summary()
    except Exception as exc:
        return Check("Device routing", WARN, f"could not compute routing: {exc}")
    return Check("Device routing", OK,
                 f"LLM→{summary['llm']}  embeddings→{summary['embeddings']}  "
                 f"whisper→{summary['whisper']}")


def check_local_genai() -> Check:
    """Can THIS machine run OpenVINO models locally (no server)?

    This is the answer to "how do I use OVAT on a Mac": openvino_genai runs
    natively on macOS/Linux/Windows CPU, which powers `ovat chat` and the
    TUI /chat screen. OVMS is only needed for the agentic tool-calling path.
    """
    try:
        import openvino_genai
        version = getattr(openvino_genai, "__version__", "installed")
    except Exception as exc:
        return Check("Local GenAI", FAIL,
                     f"openvino_genai not importable ({exc}); "
                     f"'ovat chat' and TUI /chat need it")
    return Check("Local GenAI", OK,
                 f"openvino_genai {version}: local models run here "
                 f"('ovat chat', TUI /chat; no server needed)")


def check_ovms_serving(config_binary: str | None = None) -> Check:
    """Is the OVMS serving path available HERE, and if not, why exactly.

    Platform-aware on purpose: on macOS OVMS simply does not exist, so
    saying "not on PATH" would send the user hunting for a binary they can
    never install. Elsewhere the locator's answer (config → OVAT_OVMS →
    PATH → known folders) tells them precisely how to fix it.
    """
    if sys.platform == "darwin":
        return Check("OVMS serving", WARN,
                     "OVMS does not run on macOS; develop + chat locally "
                     "here; run 'ovat serve'/'ovat run' on an AI PC or "
                     "Linux/Windows box")
    from ovat.core.ovms_locator import find_ovms
    path, how = find_ovms(config_binary)
    if path:
        return Check("OVMS serving", OK, f"ovms via {how}: {path}")
    return Check("OVMS serving", WARN,
                 f"ovms {how}; set model.ovms_binary in workflow.yml or "
                 f"the OVAT_OVMS env var to its folder")


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

    # The stub trap: search_docs is declared but no rag: block exists, so at
    # runtime the tool answers with obviously fake "[stub]" text. Legal config
    # (wiring tests rely on it), but a user should hear about it up front.
    tool_names = [t.name for t in cfg.tools]
    if "search_docs" in tool_names and cfg.rag is None:
        checks.append(Check(
            "search_docs mode", WARN,
            "declared as a tool but no rag: section; it will return stub "
            "text, not real retrieval. Add a rag: block and run 'ovat index'.",
        ))

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
    # If a config is given, its ovms_binary should inform the serving check.
    # Load it leniently here; a broken config is reported by check_config.
    config_binary = None
    if config_path:
        try:
            config_binary = load_workflow(config_path).model.ovms_binary
        except Exception:
            pass
    checks = [
        check_python(),
        check_core_deps(),
        check_local_genai(),
        *check_engines(),
        check_devices(),
        check_device_routing(),
        check_ovms_serving(config_binary),
    ]
    if config_path:
        checks.extend(check_config(config_path))
    return checks
