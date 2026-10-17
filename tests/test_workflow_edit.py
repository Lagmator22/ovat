# tests/test_workflow_edit.py
"""The files the agent app writes: workflow.yml (from /config), the UI
preferences, and the "always allow" rules.

Every one of them is written atomically; the workflow keeps its comments and
is validated whole before a byte goes to disk; the rules are owner-only and
only a session with a broker ever reads them back.
"""
import json
import os
import stat
import threading
import time

import pytest

pytest.importorskip("ruamel.yaml")

from ovat.agent.events import EventBus  # noqa: E402
from ovat.agent.permissions import (ALWAYS, ApprovalBroker,  # noqa: E402
                                    gate)
from ovat.agentic import atomic, prefs, rules  # noqa: E402
from ovat.agentic import session as s  # noqa: E402
from ovat.agentic import workflow_edit as edit  # noqa: E402
from ovat.config.workflow import WorkflowConfig, load_workflow  # noqa: E402

WORKFLOW = """\
# My agent. Keep this comment.
model:
  name: Qwen3-8B-int4-ov   # the model I like
  device: GPU
tools:
  - name: search_docs      # my notes
    type: builtin
subagents:
  - name: researcher
    description: reads docs
    api:
      base_url: https://example.com/v1
      model: big
agent:
  type: native
"""


@pytest.fixture
def workflow(tmp_path):
    path = tmp_path / "w.yml"
    path.write_text(WORKFLOW, encoding="utf-8")
    return str(path)


def _setting(path, name):
    return next(c.setting for c in edit.read_settings(path)
                if c.setting.name == name)


# ---- reading --------------------------------------------------------------------

def test_the_settings_say_what_the_file_sets_and_what_is_default(workflow):
    found = {c.setting.name: c for c in edit.read_settings(workflow)}
    assert found["device"].value == "GPU" and found["device"].in_file
    assert found["temperature"].value == 0.0 and not found["temperature"].in_file
    assert found["max_tokens"].default == 4096
    # a tool and a subagent each get a permission setting, with the default
    # the factory would give it (an api subagent asks)
    assert found["search_docs.permission"].default == "allow"
    assert found["researcher.permission"].default == "ask"
    assert found["researcher.permission"].setting.pointer == \
        "/subagents/researcher/permission"


def test_an_unreadable_workflow_is_one_sentence(tmp_path):
    with pytest.raises(edit.EditError) as err:
        edit.read_settings(str(tmp_path / "missing.yml"))
    assert "missing.yml" in str(err.value)


# ---- writing --------------------------------------------------------------------

def test_a_change_is_written_and_the_comments_stay(workflow):
    edit.apply_changes(workflow, [
        (_setting(workflow, "device"), "NPU"),
        (_setting(workflow, "temperature"), 0.5),
        (_setting(workflow, "search_docs.permission"), "ask")])
    text = open(workflow, encoding="utf-8").read()
    assert "# My agent. Keep this comment." in text
    assert "# the model I like" in text and "# my notes" in text
    config = load_workflow(workflow)
    assert config.model.device == "NPU" and config.model.temperature == 0.5
    assert config.tools[0].permission == "ask"
    # the list keeps the style of examples/: "  - name:"
    assert "\n  - name: search_docs" in text


def test_reset_takes_the_key_out_of_the_file(workflow):
    edit.apply_changes(workflow, [(_setting(workflow, "temperature"), 0.5)])
    edit.apply_changes(workflow, [(_setting(workflow, "temperature"),
                                   edit.REMOVE)])
    assert "temperature" not in open(workflow, encoding="utf-8").read()


def test_an_invalid_result_is_refused_with_the_message_and_the_file_is_untouched(
        workflow):
    before = open(workflow, "rb").read()
    with pytest.raises(edit.EditError) as err:
        edit.apply_changes(workflow, [(_setting(workflow, "top_p"), 3.0)])
    assert "model.top_p" in str(err.value)
    assert "less than or equal to 1" in str(err.value)
    assert open(workflow, "rb").read() == before


def test_one_bad_change_in_a_batch_writes_none_of_them(workflow):
    before = open(workflow, "rb").read()
    with pytest.raises(edit.EditError):
        edit.apply_changes(workflow, [
            (_setting(workflow, "temperature"), 0.5),
            (_setting(workflow, "max_tokens"), -4)])
    assert open(workflow, "rb").read() == before


def test_a_value_of_the_wrong_type_is_refused_before_it_is_saved(workflow):
    with pytest.raises(edit.EditError) as err:
        edit.apply_changes(workflow, [(_setting(workflow, "max_tokens"),
                                       "lots")])
    assert "whole number" in str(err.value)
    with pytest.raises(edit.EditError) as err:
        edit.apply_changes(workflow, [(_setting(workflow, "device"), "TPU")])
    assert "CPU, GPU, NPU" in str(err.value)


def test_empty_text_clears_a_text_setting(workflow):
    edit.apply_changes(workflow, [(_setting(workflow, "system_prompt"),
                                   "Be brief.")])
    assert load_workflow(workflow).agent.system_prompt == "Be brief."
    edit.apply_changes(workflow, [(_setting(workflow, "system_prompt"), "")])
    assert load_workflow(workflow).agent.system_prompt is None


def test_the_write_is_atomic_and_keeps_the_files_mode(workflow, monkeypatch):
    os.chmod(workflow, 0o640)
    seen = {}
    real = os.replace

    def spy(src, dst):
        seen["temp_existed"] = os.path.exists(src)
        seen["old_intact"] = open(dst, encoding="utf-8").read() == WORKFLOW
        return real(src, dst)
    monkeypatch.setattr(os, "replace", spy)
    edit.apply_changes(workflow, [(_setting(workflow, "device"), "CPU")])
    assert seen == {"temp_existed": True, "old_intact": True}
    assert not os.path.exists(workflow + ".tmp")
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(workflow).st_mode) == 0o640


def test_a_failed_write_leaves_the_old_file_and_no_temp_file(
        workflow, monkeypatch):
    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(edit.EditError) as err:
        edit.apply_changes(workflow, [(_setting(workflow, "device"), "CPU")])
    assert "disk full" in str(err.value)
    assert open(workflow, encoding="utf-8").read() == WORKFLOW
    assert not os.path.exists(workflow + ".tmp")


def test_without_ruamel_a_write_says_what_to_install(workflow, monkeypatch):
    import builtins
    real = builtins.__import__

    def refuse(name, *args, **kw):
        if name.startswith("ruamel"):
            raise ImportError(name)
        return real(name, *args, **kw)
    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(edit.EditError) as err:
        edit.apply_changes(workflow, [(_setting(workflow, "device"), "CPU")])
    assert "ruamel.yaml" in str(err.value)
    assert open(workflow, encoding="utf-8").read() == WORKFLOW


def test_every_setting_is_a_field_the_schema_has():
    """A setting that named a field the schema lacks would fail every write;
    this keeps the list honest when the schema changes."""
    config = WorkflowConfig(model={"name": "m"})
    for setting in edit.settings_for(config):
        if setting.item is None:
            section = getattr(config, setting.section)
            assert setting.key in type(section).model_fields, setting.name
    from ovat.agent.factory import AGENT_TYPES
    engine = next(x for x in edit.settings_for(config) if x.name == "engine")
    assert engine.choices == tuple(AGENT_TYPES)


# ---- the UI preferences ---------------------------------------------------------

def test_the_theme_and_the_launchers_paths_share_one_file(tmp_path):
    from ovat.cli.chat_screen import load_prefs, save_prefs
    cwd = str(tmp_path)
    save_prefs(cwd, "cfg.yml", "models/llm")
    prefs.update(cwd, theme="nord")
    # each writer changes only its own keys
    assert load_prefs(cwd) == {"config": "cfg.yml", "model_path": "models/llm",
                               "theme": "nord"}
    save_prefs(cwd, "other.yml", "models/llm")
    assert prefs.load(cwd)["theme"] == "nord"
    prefs.update(cwd, theme=None)
    assert "theme" not in prefs.load(cwd)
    assert prefs.prefs_path(cwd).name == "chat_prefs.json"


def test_odd_preferences_read_as_none(tmp_path):
    path = prefs.prefs_path(str(tmp_path))
    path.parent.mkdir(parents=True)
    path.write_text("[1, 2]")
    assert prefs.load(str(tmp_path)) == {}
    path.write_text("not json")
    assert prefs.load(str(tmp_path)) == {}


def test_atomic_write_leaves_the_old_text_when_it_fails(tmp_path, monkeypatch):
    target = tmp_path / "a" / "x.txt"
    atomic.write_text(target, "old")

    def boom(src, dst):
        raise OSError("no")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic.write_text(target, "new")
    assert target.read_text() == "old"
    assert [p.name for p in target.parent.iterdir()] == ["x.txt"]


# ---- "always allow" -------------------------------------------------------------

def _ask(broker, tool="describe_image"):
    result = {}

    def go():
        result["ok"] = broker(tool, {"path": "a.png"}, "")
    thread = threading.Thread(target=go, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not broker.pending():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return thread, result


def test_always_allow_is_saved_owner_only_and_skips_the_question(tmp_path):
    cwd = str(tmp_path)
    saved = []
    broker = ApprovalBroker(on_request=lambda r: None,
                            on_always=lambda tool: (saved.append(tool),
                                                    rules.add(cwd, tool)))
    thread, result = _ask(broker)
    broker.resolve(broker.pending()[0].id, ALWAYS)
    thread.join(5)
    assert result["ok"] is True and saved == ["describe_image"]
    assert rules.load(cwd) == ["describe_image"]
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(rules.rules_path(cwd)).st_mode) == 0o600
    assert broker("describe_image", {}, "") is True      # no second question
    # a new session's broker, given the saved list, never asks either
    fresh = ApprovalBroker(on_request=pytest.fail,
                           always=rules.load(cwd))
    assert fresh("describe_image", {}, "") is True
    # and a new CONVERSATION does not forget a rule
    broker.forget_session_grants()
    assert broker("describe_image", {}, "") is True


def test_adding_a_rule_twice_keeps_one(tmp_path):
    rules.add(str(tmp_path), "a")
    rules.add(str(tmp_path), "a")
    rules.add(str(tmp_path), "b")
    assert rules.load(str(tmp_path)) == ["a", "b"]


def test_a_hand_edited_rules_file_never_stops_the_app(tmp_path):
    path = rules.rules_path(str(tmp_path))
    path.parent.mkdir(parents=True)
    for text in ("nope", "[]", '{"always_allow": "x"}',
                 '{"always_allow": [1, "ok"]}'):
        path.write_text(text)
        assert rules.load(str(tmp_path)) in ([], ["ok"])


def test_headless_still_says_no_whatever_the_rules_file_holds(tmp_path):
    """The rules reach a broker and only a broker. A bus with no approver
    (ovat run, a bench worker, an A2A server) answers an `ask` tool with the
    same sentence as before."""
    rules.add(str(tmp_path), "describe_image")
    bus = EventBus()
    assert getattr(bus, "approver", None) is None
    said = gate(bus, "describe_image", {}, "ask")
    assert said.startswith("Error: describe_image needs a person's approval")


def test_a_session_preloads_the_rules_and_saves_a_new_one(tmp_path):
    cwd = str(tmp_path)
    rules.add(cwd, "delegate")
    seen = []
    session = s.AgentSession("w.yml", seen.append, cwd=cwd,
                             load=lambda p: None, build=lambda c: None,
                             close=lambda a: None)
    session.broker.timeout = 0.1
    assert session.broker("delegate", {}, "") is True       # preloaded
    thread, result = _ask(session.broker, "describe_image")
    session.broker.resolve(session.broker.pending()[0].id, ALWAYS)
    thread.join(5)
    assert rules.load(cwd) == ["delegate", "describe_image"]


def test_a_rule_that_cannot_be_saved_is_said_and_lasts_for_the_session(
        tmp_path, monkeypatch):
    def full(cwd, tool):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(rules, "add", full)
    seen = []
    session = s.AgentSession("w.yml", seen.append, cwd=str(tmp_path),
                             load=lambda p: None, build=lambda c: None,
                             close=lambda a: None)
    thread, result = _ask(session.broker)
    session.broker.resolve(session.broker.pending()[0].id, ALWAYS)
    thread.join(5)
    assert result["ok"] is True
    assert session.broker("describe_image", {}, "") is True
    notice = next(e for e in seen if isinstance(e, s.Notice))
    assert notice.level == "warn" and "No space left" in notice.text
    assert "this session only" in notice.text


@pytest.mark.skipif(os.name != "posix", reason="Unix file mode and symlink checks")
def test_atomic_write_ignores_a_preplanted_temp_symlink_and_keeps_files_private(tmp_path):
    target = tmp_path / "permissions.json"
    victim = tmp_path / "owner.txt"
    victim.write_text("keep me")
    predictable = target.with_suffix(".json.tmp")
    predictable.symlink_to(victim)
    target.write_text("old")
    target.chmod(0o644)
    atomic.write_text(target, "private")
    assert victim.read_text() == "keep me"
    assert predictable.is_symlink()
    assert target.read_text() == "private"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_coding_permission_defaults_match_the_built_tools(tmp_path):
    path = tmp_path / "coding.yml"
    path.write_text("model: {name: m}\ntools: [{name: shell}, {name: read_file}, {name: web_fetch}]\n")
    found = {c.setting.name: c for c in edit.read_settings(str(path))}
    assert found["shell.permission"].default == "ask"
    assert found["shell.permission"].value == "ask"
    assert found["web_fetch.permission"].default == "ask"
    assert found["read_file.permission"].default == "allow"
