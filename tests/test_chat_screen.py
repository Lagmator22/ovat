# tests/test_chat_screen.py
"""Tests for the native TUI chat screen, headless via Textual's Pilot.

Note to myself: no real model loads here. _build_components is the seam —
I monkeypatch it to return a fake retriever and a fake streaming LLM, so the
whole screen (load → ask → stream → transcript → autosave → /save → /load →
Esc) runs in milliseconds on any machine.
"""
import asyncio
import os

import pytest

pytest.importorskip("textual")

from textual.widgets import Input, RichLog

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


def test_chat_without_a_known_model_path_prints_usage(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                     # no prefs saved here yet

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt", Input)
            inp.value = "/chat"
            await pilot.press("enter")
            await pilot.pause()
            log = app.query_one("#output", RichLog)
            text = "\n".join(str(line) for line in log.lines)
            assert "usage: /chat" in text
    _run(scenario())


def test_prefs_round_trip(tmp_path):
    assert load_prefs(str(tmp_path)) == {}
    save_prefs(str(tmp_path), "cfg.yml", "models/llm")
    prefs = load_prefs(str(tmp_path))
    assert prefs == {"config": "cfg.yml", "model_path": "models/llm"}
