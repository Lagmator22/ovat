# tests/test_chat_screen.py
"""Tests for the native TUI chat screen, headless via Textual's Pilot.

Note to myself: no real model loads here. _build_components is the seam;
I monkeypatch it to return a fake retriever and a fake streaming LLM, so the
whole screen (load → ask → stream → transcript → autosave → /save → /load →
Esc) runs in milliseconds on any machine.
"""
import asyncio
import os

import pytest

pytest.importorskip("textual")

from textual.widgets import Input, RichLog, Static

from ovat.cli import chat_screen
from ovat.cli.chat_screen import ChatScreen, load_prefs, save_prefs
from ovat.cli.tui import OvatTUI
from ovat.config.workflow import WorkflowConfig


def _run(coro):
    asyncio.run(coro)


class FakeRetriever:
    def retrieve(self, query, top_k=5):
        return [{"text": "Q3 revenue grew", "source": "finance.md", "distance": 0.1}]


class FakeLLM:
    def chat(self, messages, tools=None, on_token=None):
        if on_token is not None:
            for token in ("AN", "SW", "ER"):
                if on_token(token):
                    break
        return {"finish_reason": "stop", "content": "ANSWER", "tool_calls": None}


def _fake_components(config_path, model_path, max_tokens=256):
    cfg = WorkflowConfig(model={"name": "m"})
    return cfg, FakeRetriever(), FakeLLM()


def _log_text(screen) -> str:
    log = screen.query_one("#chat-log", RichLog)
    return "\n".join(str(line) for line in log.lines)


async def _push_ready_screen(app, pilot, tmp_path):
    screen = ChatScreen("w.yml", "model-dir", cwd=str(tmp_path))
    app.push_screen(screen)
    await app.workers.wait_for_complete()      # the load worker
    await pilot.pause()
    return screen


def test_full_turn_streams_answers_and_autosaves(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            inp = screen.query_one("#chat-input", Input)
            inp.value = "how did revenue do?"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            text = _log_text(screen)
            assert "you › how did revenue do?" in text
            assert "ovat › ANSWER" in text
            assert "sources: finance.md" in text
            # Every turn autosaves, so a crash never loses the conversation.
            assert os.path.exists(tmp_path / ".ovat" / "sessions" / "last.json")
    _run(scenario())


def test_stream_tail_clips_the_live_line():
    """The live line shows the TAIL, so it cannot grow without bound."""
    long_tail = chat_screen._stream_tail(["word "] * 500)
    assert len(long_tail.plain) <= chat_screen.STREAM_TAIL_CHARS + len("ovat › …")
    assert long_tail.plain.startswith("ovat › …")     # marked as clipped
    assert long_tail.plain.endswith("word ")          # and it is the END we kept
    # A short answer is shown whole, with no ellipsis.
    assert chat_screen._stream_tail(["hi ", "there"]).plain == "ovat › hi there"


def test_live_stream_line_never_squeezes_out_the_log_or_input(monkeypatch, tmp_path):
    """The regression: a long answer used to push the input off the screen.

    Measured before the fix on a 30-row screen: at 400 streamed tokens
    #chat-stream had grown to 27 rows, #chat-log was down to 0, and the input
    sat at y=31, i.e. below the bottom edge. max-height plus the tail render
    keep the live line small enough that the transcript and input survive.
    """
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test(size=(100, 30)) as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            log = screen.query_one("#chat-log", RichLog)
            stream = screen.query_one("#chat-stream", Static)
            inp = screen.query_one("#chat-input", Input)

            stream.update(chat_screen._stream_tail(["token "] * 400))
            await pilot.pause()
            await pilot.pause()          # let the layout settle before measuring

            assert stream.size.height <= 6                  # capped
            assert log.size.height >= 10                    # transcript survives
            assert inp.region.y + inp.size.height <= app.size.height   # still visible
    _run(scenario())


def test_save_and_load_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            inp = screen.query_one("#chat-input", Input)
            inp.value = "remember me"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            inp.value = "/save demo"
            await pilot.press("enter")
            await pilot.pause()
            assert os.path.exists(tmp_path / ".ovat" / "sessions" / "demo.json")

            # A brand-new screen loads the same conversation back.
            app.pop_screen()
            fresh = await _push_ready_screen(app, pilot, tmp_path)
            fresh.query_one("#chat-input", Input).value = "/load demo"
            await pilot.press("enter")
            await pilot.pause()
            assert "remember me" in _log_text(fresh)
    _run(scenario())


def test_escape_when_idle_returns_to_the_launcher(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            assert app.screen is screen
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is not screen         # back on the launcher
    _run(scenario())


def test_load_failure_reports_and_offers_the_way_back(monkeypatch, tmp_path):
    def broken(config_path, model_path, max_tokens=256):
        raise ValueError("this workflow has no rag: section")
    monkeypatch.setattr(chat_screen, "_build_components", broken)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            assert "Could not start chat" in _log_text(screen)
            assert "no rag: section" in _log_text(screen)
    _run(scenario())


def test_chat_with_no_model_and_none_found_explains_the_fix(monkeypatch, tmp_path):
    import ovat.core.model_scout as scout
    monkeypatch.chdir(tmp_path)                     # no prefs saved here yet
    monkeypatch.setattr(scout, "pick_chat_llm", lambda: (None, []))
    monkeypatch.setattr(scout, "find_models", lambda kind=None: [
        {"name": "Qwen2-VL-2B", "path": "/x", "kind": "vlm", "why": ""}])

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = "/chat"
            await pilot.press("enter")
            await pilot.pause()
            log = app.query_one("#output", RichLog)
            text = "\n".join(str(line) for line in log.lines)
            assert "No local text LLM found" in text
            assert "Qwen2-VL-2B (vlm)" in text       # names what it DID find
            assert "OVAT_MODELS" in text             # and the fix
    _run(scenario())


def test_chat_with_no_model_auto_detects_one(monkeypatch, tmp_path):
    import ovat.core.model_scout as scout
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)
    llama = {"name": "Llama-3B-Instruct", "path": str(tmp_path / "llama"),
             "kind": "llm", "why": ""}
    monkeypatch.setattr(scout, "pick_chat_llm", lambda: (llama, [llama]))

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = "/chat"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)      # it just worked
            log = app.screen_stack[0].query_one("#output", RichLog)
            text = "\n".join(str(line) for line in log.lines)
            assert "auto-detected local LLM: Llama-3B-Instruct" in text
    _run(scenario())


def test_build_components_refuses_a_vision_model_before_loading(tmp_path):
    # The real-user bug: a Qwen2-VL folder given as the chat LLM used to load
    # and then explode at generate time with a C++ tensor-port traceback.
    import json

    import pytest as _pytest
    vlm = tmp_path / "Qwen2-VL-2B-Instruct-INT4"
    vlm.mkdir()
    (vlm / "openvino_language_model.xml").write_text("")
    (vlm / "config.json").write_text(json.dumps({"model_type": "qwen2_vl"}))
    cfg = tmp_path / "w.yml"
    cfg.write_text("model:\n  name: m\nrag:\n  retriever:\n    db_path: ':memory:'\n")

    with _pytest.raises(ValueError, match="not a text"):
        chat_screen._build_components(str(cfg), str(vlm))


def test_prefs_round_trip(tmp_path):
    assert load_prefs(str(tmp_path)) == {}
    save_prefs(str(tmp_path), "cfg.yml", "models/llm")
    prefs = load_prefs(str(tmp_path))
    assert prefs == {"config": "cfg.yml", "model_path": "models/llm"}
