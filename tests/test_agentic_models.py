# tests/test_agentic_models.py
"""The models a session can switch to, and how a switch reaches the build.

No server and no model folders are needed: the HTTP call and the folder scan
are faked, and the session builds a stand-in agent.
"""
import json
import threading

from ovat.agent.events import EventBus
from ovat.agentic import models
from ovat.agentic import session as s
from ovat.config.workflow import WorkflowConfig


# conftest replaces models.served_models for every test (no test may depend
# on what listens on this machine's port 8000); these keep the real one.
real_served_models = models.served_models


class _Reply:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _serving(monkeypatch, payload, asked=None):
    class Opener:
        def open(self, url, timeout=None):
            if asked is not None:
                asked.append(url)
            return _Reply(payload)
    monkeypatch.setattr(models.urllib.request, "build_opener",
                        lambda *handlers: Opener())


def test_served_models_reads_the_ids_the_server_lists(monkeypatch):
    asked = []
    _serving(monkeypatch, {"data": [{"id": "Qwen3-8B"}, {"id": "bge"},
                                    {"object": "model"}]}, asked)
    assert real_served_models("http://localhost:8000/v3/") == [
        "Qwen3-8B", "bge"]
    assert asked == ["http://localhost:8000/v3/models"]


def test_served_models_is_empty_when_the_server_cannot_be_asked(monkeypatch):
    def refuse(*handlers):
        raise OSError("refused")
    monkeypatch.setattr(models.urllib.request, "build_opener", refuse)
    assert real_served_models("http://localhost:8000/v3") == []
    _serving(monkeypatch, ["not", "an", "object"])
    assert real_served_models("http://localhost:8000/v3") == []


def _config(**model):
    return WorkflowConfig(model={"name": "Qwen3-8B", **model})


def test_an_ovms_session_is_offered_only_what_the_server_serves(monkeypatch):
    monkeypatch.setattr(models, "served_models",
                        lambda url: ["Qwen3-8B", "Qwen3.5-4B"])
    assert [c.id for c in models.choices(_config())] == [
        "Qwen3-8B", "Qwen3.5-4B"]


def test_a_server_that_lists_nothing_leaves_the_workflows_own_model():
    # the autouse fixture makes served_models answer []: no server
    assert [(c.id, c.label) for c in models.choices(_config())] == [
        ("Qwen3-8B", "Qwen3-8B")]


def test_a_local_session_is_offered_the_chat_models_on_disk(monkeypatch):
    found = [{"name": "Llama-3B", "path": "/m/Llama-3B", "kind": "llm"},
             {"name": "Qwen3.5-0.8B", "path": "/m/Qwen3.5-0.8B",
              "kind": "unified"},
             {"name": "bge-small", "path": "/m/bge-small",
              "kind": "embeddings"},
             {"name": "whisper", "path": "/m/whisper", "kind": "whisper"}]
    monkeypatch.setattr(models.model_scout, "find_models",
                        lambda kind=None, extra_roots=None: found)
    got = models.choices(_config(provider="genai", name="/m/Llama-3B"))
    assert [(c.id, c.label) for c in got] == [
        ("/m/Llama-3B", "Llama-3B"), ("/m/Qwen3.5-0.8B", "Qwen3.5-0.8B")]


def test_a_local_model_outside_the_scanned_folders_stays_a_choice(monkeypatch):
    monkeypatch.setattr(models.model_scout, "find_models",
                        lambda kind=None, extra_roots=None: [])
    got = models.choices(_config(provider="genai", name="/elsewhere/Phi-3"))
    assert [(c.id, c.label) for c in got] == [
        ("/elsewhere/Phi-3", "Phi-3")]


# ---- how the session builds with a pick -----------------------------------------

class _Agent:
    def __init__(self):
        self.events = EventBus()
        self.tools = {}


class _Sink:
    def __init__(self):
        self.events = []
        self.changed = threading.Condition()

    def __call__(self, event):
        with self.changed:
            self.events.append(event)
            self.changed.notify_all()

    def wait(self, kind, count):
        with self.changed:
            assert self.changed.wait_for(
                lambda: sum(isinstance(e, kind) for e in self.events) >= count,
                timeout=10)


def _session(fail_on=None):
    sink, built = _Sink(), []

    def build(config):
        if fail_on and config.model.name == fail_on:
            raise RuntimeError("cannot load that")
        built.append(config)
        return _Agent()

    session = s.AgentSession(
        "w.yml", sink, build=build, close=lambda a: None,
        load=lambda path: WorkflowConfig(model={"name": "Qwen3-8B"}))
    return session, sink, built


def _ready(session, sink, count):
    sink.wait(s.Ready, count)
    return session.info


def test_a_pick_reaches_the_config_and_the_banner_facts():
    session, sink, built = _session()
    session.start()
    assert _ready(session, sink, 1)["model_id"] == "Qwen3-8B"
    session.switch_model(model="Qwen3.5-4B", enable_thinking=True)
    info = _ready(session, sink, 2)
    assert built[1].model.name == "Qwen3.5-4B"
    assert built[1].model.enable_thinking is True
    assert (info["model_id"], info["thinking"]) == ("Qwen3.5-4B", True)
    # the file's own model is remembered, to go back to
    assert info["default_model_id"] == "Qwen3-8B"


def test_the_pick_survives_a_rebuild_for_another_engine_and_a_clear():
    session, sink, built = _session()
    session.start()
    sink.wait(s.Ready, 1)
    session.switch_model(model="Qwen3.5-4B")
    sink.wait(s.Ready, 2)
    session.start("react")
    sink.wait(s.Ready, 3)
    assert built[2].model.name == "Qwen3.5-4B"
    assert built[2].agent.type == "react"
    # and a model pick keeps the engine that is running
    session.switch_model(enable_thinking=False)
    sink.wait(s.Ready, 4)
    assert built[3].agent.type == "react"
    assert built[3].model.name == "Qwen3.5-4B"


def test_the_empty_pick_goes_back_to_the_workflows_own_model():
    session, sink, built = _session()
    session.start()
    sink.wait(s.Ready, 1)
    session.switch_model(model="Qwen3.5-4B")
    sink.wait(s.Ready, 2)
    session.switch_model(model="")
    sink.wait(s.Ready, 3)
    assert built[2].model.name == "Qwen3-8B"


def test_a_pick_that_cannot_build_is_not_kept():
    session, sink, built = _session(fail_on="broken")
    session.start()
    sink.wait(s.Ready, 1)
    session.switch_model(model="broken")
    sink.wait(s.BuildFailed, 1)
    session.start()                      # any later rebuild
    sink.wait(s.Ready, 2)
    assert built[-1].model.name == "Qwen3-8B"


def test_another_workflow_starts_without_the_picks():
    session, sink, built = _session()
    session.start()
    sink.wait(s.Ready, 1)
    session.switch_model(model="Qwen3.5-4B")
    sink.wait(s.Ready, 2)
    session.switch_workflow("other.yml")
    sink.wait(s.Ready, 3)
    assert built[2].model.name == "Qwen3-8B"


def test_a_settings_reload_reads_the_workflows_new_engine(monkeypatch):
    session, sink, built = _session()
    session.start()
    _ready(session, sink, 1)
    monkeypatch.setattr(session, "_load", lambda path: WorkflowConfig(
        model={"name": "Qwen3-8B"}, agent={"type": "react"}))
    session.switch_model(follow_workflow=True)
    _ready(session, sink, 2)
    assert built[-1].agent.type == "react"
