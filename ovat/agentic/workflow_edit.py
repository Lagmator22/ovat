# ovat/agentic/workflow_edit.py
"""Editing workflow.yml from the settings screen, without breaking it.

Note to myself: the rules of this file are the ones the owner set.

- Only fields OVAT really has. SETTINGS below is the whole list: the model's
  sampling and device, the engine, a tool's or subagent's permission. Anything
  else is refused by the caller ("not used by OVAT").
- The user's comments survive. The file is read and written back with
  ruamel.yaml in round-trip mode (the plain yaml module drops every comment).
  ruamel lives in the `tui` extra only; without it a write says so.
- The WHOLE result is validated with WorkflowConfig BEFORE anything is written.
  A bad value raises EditError with pydantic's own message and the file is
  not touched.
- The write is atomic (ovat/agentic/atomic.py), and the file keeps its mode.

Nothing here imports textual. Reading (`read_settings`) needs only PyYAML.
"""
from __future__ import annotations

import io
import os
import stat
from dataclasses import dataclass
from typing import Any

import yaml
from pydantic import ValidationError

from ovat.agentic import atomic
from ovat.config.workflow import WorkflowConfig


class EditError(Exception):
    """The edit cannot be made. The message is one plain sentence for the
    person; nothing was written."""


# The kinds the settings screen can edit.
BOOL, ENUM, INT, FLOAT, STR = "bool", "enum", "int", "float", "str"
REMOVE = object()           # "back to the default": the key leaves the file


@dataclass(frozen=True)
class Setting:
    name: str               # what the screen shows
    pointer: str            # what a config/write op addresses
    section: str            # model | agent | tools | subagents
    key: str                # the field in that section
    kind: str
    description: str
    choices: tuple[str, ...] = ()
    popular: bool = False
    item: str | None = None  # a tool's or subagent's name (a list entry)


def _fixed() -> list[Setting]:
    from ovat.agent.factory import AGENT_TYPES
    return [
        Setting("active_model", "/active_model", "model", "name", STR,
                "The model this workflow runs. Changing it here or with "
                "/model saves it in the workflow file.",
                popular=True),
        Setting("device", "/model/device", "model", "device", ENUM,
                "Where OVMS runs the model: CPU, GPU or NPU.",
                choices=("CPU", "GPU", "NPU"), popular=True),
        Setting("engine", "/agent/type", "agent", "type", ENUM,
                "Which agent engine runs the tool-calling loop.",
                choices=tuple(AGENT_TYPES), popular=True),
        Setting("enable_thinking", "/model/enable_thinking", "model",
                "enable_thinking", BOOL,
                "Let a Qwen3 or Qwen3.5 model reason before it answers. "
                "Unset: the chat template decides.", popular=True),
        Setting("temperature", "/model/temperature", "model", "temperature",
                FLOAT, "Sampling temperature. 0 is the most likely token."),
        Setting("max_tokens", "/model/max_tokens", "model", "max_tokens", INT,
                "The longest one reply may be, in tokens."),
        Setting("top_p", "/model/top_p", "model", "top_p", FLOAT,
                "Nucleus sampling, above 0 up to 1."),
        Setting("top_k", "/model/top_k", "model", "top_k", INT,
                "Sample only from the k likeliest tokens (-1: all)."),
        Setting("min_p", "/model/min_p", "model", "min_p", FLOAT,
                "Drop tokens less likely than this share of the best one."),
        Setting("presence_penalty", "/model/presence_penalty", "model",
                "presence_penalty", FLOAT,
                "Between -2 and 2; above 0 pushes the model to new topics."),
        Setting("seed", "/model/seed", "model", "seed", INT,
                "A fixed seed for sampling."),
        Setting("request_timeout", "/model/request_timeout", "model",
                "request_timeout", FLOAT,
                "Seconds to wait for OVMS before giving up on one request."),
        Setting("context_window", "/model/context_window", "model",
                "context_window", INT,
                "The model's context size in tokens, when it cannot be read "
                "from the model folder."),
        Setting("max_iterations", "/agent/max_iterations", "agent",
                "max_iterations", INT,
                "The most model calls one question may take."),
        Setting("system_prompt", "/agent/system_prompt", "agent",
                "system_prompt", STR, "The agent's persona and rules."),
    ]


def _permission(section: str, name: str) -> Setting:
    return Setting(
        f"{name}.permission", f"/{section}/{name}/permission", section,
        "permission", ENUM,
        f"Whether {name} may run: allow, ask a person first, or deny.",
        choices=("allow", "ask", "deny"), item=name)


def settings_for(config: WorkflowConfig) -> list[Setting]:
    """Every setting this workflow has: the fixed ones, then one permission
    per tool and subagent the file names."""
    return [*_fixed(),
            *(_permission("tools", t.name) for t in config.tools),
            *(_permission("subagents", s.name) for s in config.subagents)]


def find_setting(config: WorkflowConfig, pointer: str) -> Setting | None:
    return next((s for s in settings_for(config) if s.pointer == pointer),
                None)


# ---- reading ---------------------------------------------------------------

@dataclass(frozen=True)
class Current:
    setting: Setting
    value: Any              # what the workflow runs with
    default: Any            # what it would run with if the file said nothing
    in_file: bool           # the file sets it


def _section_of(config: WorkflowConfig, setting: Setting):
    if setting.section in ("model", "agent"):
        return getattr(config, setting.section)
    entries = getattr(config, setting.section)
    return next(e for e in entries if e.name == setting.item)


def _default_of(setting: Setting, section) -> Any:
    if setting.key == "permission":
        from ovat.agent.permissions import default_permission
        if setting.section == "tools":
            return default_permission(section.type, section.url, section.name)
        from ovat.agent.subagents import kind_of
        return default_permission(
            kind_of(section), section.a2a.url if section.a2a else None)
    return type(section).model_fields[setting.key].get_default(
        call_default_factory=True)


def _raw_entry(raw: dict, setting: Setting) -> dict:
    node = raw.get(setting.section)
    if setting.item is None:
        return node if isinstance(node, dict) else {}
    for entry in node if isinstance(node, list) else []:
        if isinstance(entry, dict) and entry.get("name") == setting.item:
            return entry
    return {}


def read_settings(path: str) -> list[Current]:
    """The settings of the workflow at `path` with what each is now. Raises
    EditError when the file cannot be read or does not validate."""
    try:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        # The editor already needs the raw mapping to report explicit keys;
        # validate that same read through the workflow's own schema.
        config = WorkflowConfig.model_validate(raw)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise EditError(f"Cannot read {path}: {_first_line(exc)}") from None
    current = []
    for setting in settings_for(config):
        section = _section_of(config, setting)
        default = _default_of(setting, section)
        value = getattr(section, setting.key)
        if setting.key == "permission" and value is None:
            value = default
        current.append(Current(
            setting, value, default,
            setting.key in _raw_entry(raw, setting)))
    return current


# ---- writing ---------------------------------------------------------------

def apply_changes(path: str, changes: list[tuple[Setting, Any]]) -> None:
    """Make these changes in the workflow at `path`: (setting, new value) or
    (setting, REMOVE) to go back to the default. Raises EditError, with
    nothing written, if the result would not be a valid workflow."""
    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.error import YAMLError
    except ImportError:
        raise EditError("Saving settings needs ruamel.yaml, which keeps your "
                        "comments. Install it with: "
                        "pip install \"ovat[tui]\"") from None
    yml = YAML(typ="rt")
    yml.preserve_quotes = True
    yml.width = 4096                    # do not re-wrap a long prompt line
    yml.indent(mapping=2, sequence=4, offset=2)     # "  - name:", as in examples/
    try:
        with open(path, encoding="utf-8") as handle:
            original = handle.read()
        doc = yml.load(original)
    except (OSError, YAMLError) as exc:
        raise EditError(f"Cannot read {path}: {_first_line(exc)}") from None
    if not isinstance(doc, dict):
        raise EditError(f"{path} is not a workflow (a mapping of "
                        f"key: value pairs).")
    for setting, value in changes:
        _change(doc, setting, value)
    out = io.StringIO()
    yml.dump(doc, out)
    text = out.getvalue()
    _validate(text)
    if text == original:
        return
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
        atomic.write_text(path, text, mode=mode)
    except OSError as exc:
        raise EditError(f"Cannot write {path}: {_first_line(exc)}") from None


def coerce(setting: Setting, value: Any) -> Any:
    """The value as the YAML should hold it, or EditError. The wire value is
    JSON; a string where a number belongs would otherwise be saved as a
    string and only coerced by pydantic on every later load."""
    if value is REMOVE:
        return value
    ok = {BOOL: isinstance(value, bool),
          INT: isinstance(value, int) and not isinstance(value, bool),
          FLOAT: isinstance(value, (int, float)) and not isinstance(value, bool),
          STR: isinstance(value, str),
          ENUM: isinstance(value, str)}[setting.kind]
    if not ok:
        want = {BOOL: "true or false", INT: "a whole number",
                FLOAT: "a number", STR: "text", ENUM: "one of the choices"}
        raise EditError(f"{setting.name} must be {want[setting.kind]}.")
    if setting.kind == ENUM and value not in setting.choices:
        raise EditError(f"{setting.name} must be one of: "
                        f"{', '.join(setting.choices)}.")
    if setting.kind == FLOAT:
        return float(value)
    # An empty text means "none": the key leaves the file, so the field is
    # back to its default instead of holding "".
    if setting.kind == STR and value == "":
        return REMOVE
    return value


def _change(doc, setting: Setting, value: Any) -> None:
    value = coerce(setting, value)
    node = _find_node(doc, setting, create=value is not REMOVE)
    if node is None:
        return                          # nothing to remove
    if value is REMOVE:
        node.pop(setting.key, None)
    else:
        node[setting.key] = value


def _find_node(doc, setting: Setting, *, create: bool):
    if setting.item is None:
        node = doc.get(setting.section)
        if node is None and create:
            node = doc[setting.section] = type(doc)()
        return node
    for entry in doc.get(setting.section) or []:
        if isinstance(entry, dict) and entry.get("name") == setting.item:
            return entry
    raise EditError(f"{setting.item} is not in the workflow file any more.")


def _validate(text: str) -> None:
    """The whole file, as it would be written, must still be a workflow."""
    try:
        data = yaml.safe_load(text) or {}
        WorkflowConfig(**data)
    except ValidationError as exc:
        error = exc.errors()[0]
        where = ".".join(str(part) for part in error["loc"])
        raise EditError(f"{where}: {error['msg']}") from None
    except (ValueError, TypeError, yaml.YAMLError) as exc:
        raise EditError(_first_line(exc)) from None


def _first_line(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else type(exc).__name__
