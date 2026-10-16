# ovat/tui/server/settings.py
"""The settings screen's side of the server: which settings OVAT has, what
they are now, and what a `config/write` may do.

Mirrors Vibe's vibe/app_server/_config_introspect.py (the field list the
screen draws: kind, popular, where each value is set) and _config_write.py
(the ops a write carries). The difference is where the values live. Vibe has
layers of TOML; OVAT has ONE workflow file (the layer "workflow"), the
session's temporary overrides (the layer "overrides") and the UI
preferences file (the layer "preferences", only the theme).

The file work itself is ovat/agentic/workflow_edit.py and ovat/agentic/prefs.py.
Here the wire models are only filled in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ovat.agentic import workflow_edit as edit
from ovat.tui.vibe.app_server.config import ConfigView
from ovat.tui.vibe.app_server.protocol import (ConfigFieldKind,
                                               ConfigFieldWire,
                                               ConfigLayerValueWire,
                                               ConfigWriteOpWire)
from ovat.tui.vibe.config_values import AUTO_THEME

WORKFLOW_LAYER = "workflow"
OVERRIDES_LAYER = "overrides"       # the screen calls this one "temporary"
PREFERENCES_LAYER = "preferences"
DEFAULT_LAYER = "default"
# The layers a write may name. A screen or picker edit means the file;
# temporary overrides are kept by the session, never written here.
TARGETS = [WORKFLOW_LAYER]

THEME_POINTER = "/theme"
ACTIVE_MODEL_POINTER = "/active_model"
THINKING_POINTER = "/model/enable_thinking"

_KINDS = {edit.BOOL: ConfigFieldKind.BOOL, edit.ENUM: ConfigFieldKind.ENUM,
          edit.INT: ConfigFieldKind.INT, edit.FLOAT: ConfigFieldKind.FLOAT,
          edit.STR: ConfigFieldKind.STR}


# ---- reading: config/fields/read --------------------------------------------

def fields(info: dict, workflow: str | None, theme: str,
           theme_saved: bool) -> list[ConfigFieldWire]:
    """Every setting the screen lists. A workflow that cannot be read
    contributes none (the theme is still there): `/config` then shows what
    can still be changed."""
    wires = []
    try:
        current = edit.read_settings(workflow) if workflow else []
    except edit.EditError:
        current = []
    for item in current:
        wires.append(_wire(item, info))
    wires.append(ConfigFieldWire(
        name="theme", kind=ConfigFieldKind.STR, path=THEME_POINTER,
        description="The colours of this app, one of Textual's themes. "
                    "\"auto\" is OVAT's own.",
        value=theme, popular=True,
        layer_values=[
            *([ConfigLayerValueWire(layer=PREFERENCES_LAYER, value=theme)]
              if theme_saved else []),
            ConfigLayerValueWire(layer=DEFAULT_LAYER, value=AUTO_THEME)]))
    return wires


def _wire(item: edit.Current, info: dict) -> ConfigFieldWire:
    setting = item.setting
    value = item.value
    layers = []
    # What the session runs, when a /model or /thinking pick put it
    # somewhere else than the file says.
    if setting.pointer == ACTIVE_MODEL_POINTER:
        running = info.get("model_id") or value
        if running != value:
            layers.append(ConfigLayerValueWire(layer=OVERRIDES_LAYER,
                                               value=running))
            value = running
    elif setting.pointer == THINKING_POINTER and "thinking" in info:
        if info["thinking"] != value:
            layers.append(ConfigLayerValueWire(layer=OVERRIDES_LAYER,
                                               value=info["thinking"]))
            value = info["thinking"]
    if item.in_file:
        layers.append(ConfigLayerValueWire(layer=WORKFLOW_LAYER,
                                           value=item.value))
    # The active model has no default: the file always names one.
    if setting.pointer != ACTIVE_MODEL_POINTER:
        layers.append(ConfigLayerValueWire(layer=DEFAULT_LAYER,
                                           value=item.default))
    return ConfigFieldWire(
        name=setting.name, kind=_KINDS[setting.kind], path=setting.pointer,
        description=setting.description, value=value,
        popular=setting.popular, enum_choices=list(setting.choices),
        layer_values=layers)


# ---- writing: config/write ---------------------------------------------------

@dataclass
class Plan:
    """What a batch of ops comes to. Nothing is done while planning, so a
    batch with one bad op changes nothing."""
    changes: list = field(default_factory=list)    # (Setting, value|REMOVE)
    forget: list = field(default_factory=list)     # session picks to drop
    unpin_model: bool = False   # "follow the workflow's model" without a write
    theme: str | None = None    # the new theme, "auto" when it is cleared
    failures: list = field(default_factory=list)

    @property
    def touches_workflow(self) -> bool:
        return bool(self.changes)

    @property
    def changes_the_agent(self) -> bool:
        return bool(self.changes or self.unpin_model)


def plan(ops: list[ConfigWriteOpWire], workflow: str | None,
         theme_names) -> Plan:
    """Sort the ops into file edits, a theme, and refusals. `theme_names()`
    returns the theme names that exist (Textual's, and OVAT's own)."""
    result = Plan()
    known = None
    for op in ops:
        if op.target_layer not in (None, *TARGETS):
            result.failures.append(f"Unknown layer: {op.target_layer}. "
                                   f"Settings are saved in the "
                                   f"{WORKFLOW_LAYER}.")
            continue
        remove = op.op == "remove"
        if op.path == THEME_POINTER:
            _plan_theme(result, op, remove, theme_names)
            continue
        if op.path == ACTIVE_MODEL_POINTER and (remove or op.value == ""):
            # Back to the workflow's own model: only the session's pick goes.
            result.unpin_model = True
            result.forget.append("name")
            continue
        if known is None:
            try:
                known = edit.read_settings(workflow) if workflow else None
            except edit.EditError as exc:
                result.failures.append(str(exc))
                continue
            if known is None:
                result.failures.append(
                    "There is no workflow file to save this in. Start with "
                    "`ovat agent path/to/workflow.yml`.")
                continue
        setting = next((c.setting for c in known
                        if c.setting.pointer == op.path), None)
        if setting is None:
            result.failures.append(_not_a_setting(op.path))
            continue
        try:
            value = edit.REMOVE if remove else edit.coerce(setting, op.value)
        except edit.EditError as exc:
            result.failures.append(str(exc))
            continue
        result.changes.append((setting, value))
        # The file now says what a session pick said: the pick must not
        # keep winning over it.
        if op.path == ACTIVE_MODEL_POINTER:
            result.forget.append("name")
        elif op.path == THINKING_POINTER:
            result.forget.append("enable_thinking")
    return result


def _plan_theme(result: Plan, op: ConfigWriteOpWire, remove: bool,
                theme_names) -> None:
    name = AUTO_THEME if remove else op.value
    if not isinstance(name, str) or name not in theme_names():
        result.failures.append(f"Unknown theme: {name}")
    else:
        result.theme = name


def _not_a_setting(pointer: str) -> str:
    """Vibe has about thirty settings OVAT has no use for (voice, narrator,
    other vendors' models...). They are named in views.config_view with plain
    off values; a write to one is answered, not applied."""
    name = pointer.strip("/").split("/")[0]
    if name in ConfigView.model_fields:
        return f"'{name}' is not used by OVAT, so nothing was changed."
    return f"Unknown setting: {pointer}"
