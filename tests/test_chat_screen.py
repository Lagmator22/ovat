# tests/test_chat_screen.py
"""Tests for the native TUI chat screen, headless via Textual's Pilot.

Note to myself: no real model loads here. _build_components is the seam;
I monkeypatch it to return a fake retriever and a fake streaming LLM, so the
whole screen (load → ask → stream → transcript → autosave → /save → /load →
Esc) runs in milliseconds on any machine.
"""
import asyncio
import json
import os

import pytest

pytest.importorskip("textual")

from textual.widgets import Input, OptionList, RichLog, Static
from textual.containers import VerticalScroll

from ovat.cli import chat_screen
from ovat.cli.chat_screen import ChatScreen, load_prefs, save_prefs
from ovat.cli.tui import OvatTUI
from ovat.cli.widgets import ChatInput
from ovat.config.workflow import WorkflowConfig


def _run(coro):
    asyncio.run(coro)


class FakeRetriever:
    def retrieve(self, query, top_k=5):
        return [{"text": "Q3 revenue grew", "source": "finance.md", "distance": 0.1}]

    def close(self):
        # RetrieverProvider gives every real retriever a close(); the fake
        # needs one too or it is not standing in for the contract.
        pass


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
    """Everything in the transcript, as text.

    The transcript is a VerticalScroll of message widgets now, so this reads
    each one's source: ChatMessage keeps the markdown it was given, and the
    status/reasoning lines are Statics whose renderable is their text.
    """
    view = screen.query_one("#chat-view", VerticalScroll)
    parts = []
    for child in view.children:
        text = getattr(child, "text", None)          # Prompt / Response
        if text is None:
            text = str(getattr(child, "renderable", ""))   # Reasoning / Sources
        parts.append(text)
    return "\n".join(parts)


async def _push_ready_screen(app, pilot, tmp_path):
    """Push the chat screen and wait until it is genuinely usable.

    The pause BEFORE wait_for_complete is what makes this reliable: the load
    worker is started from on_mount, so on a slow machine it does not exist
    yet when we ask to wait for it, wait_for_complete returns instantly with
    nothing to wait for, and the test then queries a screen that has not
    finished composing. On the Windows runner that surfaced as
    NoMatches("No nodes match '#chat-view'") -- the widget was real, the test
    was simply early.

    Waiting on the WIDGET rather than only on the workers is the part that
    matters: workers finishing and the DOM being ready are two different
    events, and it was the second one every failing test depended on.
    """
    screen = ChatScreen("w.yml", "model-dir", cwd=str(tmp_path))
    app.push_screen(screen)
    await pilot.pause()                        # let on_mount start the worker
    for _ in range(60):                        # bounded: never hangs the suite
        await app.workers.wait_for_complete()
        await pilot.pause()
        if screen.query("#chat-view") and not any(
                worker.is_running for worker in app.workers):
            break
    await pilot.pause()
    return screen


def test_full_turn_streams_answers_and_autosaves(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            inp = screen.query_one("#chat-input", ChatInput)
            inp.value = "how did revenue do?"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            text = _log_text(screen)
            assert "how did revenue do?" in text          # the Prompt widget
            assert "ANSWER" in text                       # the Response widget
            assert "finance.md" in text
            # Every turn autosaves, into THIS conversation's own slot rather
            # than a shared "last" that each new chat would overwrite.
            saved = list((tmp_path / ".ovat" / "sessions").glob("auto-*.json"))
            assert len(saved) == 1
            assert screen._autosave in saved[0].name
    _run(scenario())






def test_save_and_load_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            inp = screen.query_one("#chat-input", ChatInput)
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
            fresh.query_one("#chat-input", ChatInput).value = "/load demo"
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


# /tokens: retune the answer cap without a model reload

class _CapLLM(FakeLLM):
    max_new_tokens = 256


def _components_with_cap(config_path, model_path, max_tokens=256):
    cfg = WorkflowConfig(model={"name": "m"})
    return cfg, FakeRetriever(), _CapLLM()


def _send(screen, pilot, text):
    screen.query_one("#chat-input", ChatInput).value = text
    return pilot.press("enter")


def test_tokens_command_retunes_the_live_provider(monkeypatch, tmp_path):
    """The screen loaded with a 256 cap and no way to change it, so long
    answers were simply cut off. /tokens must take effect with no reload."""
    monkeypatch.setattr(chat_screen, "_build_components", _components_with_cap)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            engine = screen._engine
            assert engine.max_new_tokens == 256

            await _send(screen, pilot, "/tokens 2048")
            await pilot.pause()
            assert engine.max_new_tokens == 2048       # same object, no reload

            await _send(screen, pilot, "/tokens 0")
            await pilot.pause()
            assert engine.max_new_tokens is None       # 0 means no cap
            assert "no tokens" in _log_text(screen)
    _run(scenario())


def test_tokens_command_refuses_nonsense_without_crashing(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_screen, "_build_components", _components_with_cap)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            engine = screen._engine

            await _send(screen, pilot, "/tokens lots")
            await pilot.pause()
            assert "wants a number" in _log_text(screen)
            assert engine.max_new_tokens == 256        # unchanged

            await _send(screen, pilot, "/tokens -1")
            await pilot.pause()
            assert "cannot be negative" in _log_text(screen)
            assert engine.max_new_tokens == 256
    _run(scenario())


# /copy: last answer, last question, or the whole conversation

def _copy_scenario(monkeypatch, tmp_path, commands):
    """Run one turn, then a list of /copy commands; return what was copied."""
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)
    copied = []

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
            inp = screen.query_one("#chat-input", ChatInput)
            inp.value = "how did revenue do?"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            for command in commands:
                inp.value = command
                await pilot.press("enter")
                await pilot.pause()
            _copy_scenario.log = _log_text(screen)
    _run(scenario())
    return copied


def test_copy_takes_the_last_answer_by_default(monkeypatch, tmp_path):
    copied = _copy_scenario(monkeypatch, tmp_path, ["/copy"])
    assert copied == ["ANSWER"]
    assert "copied the last answer" in _copy_scenario.log


def test_copy_me_takes_your_last_question(monkeypatch, tmp_path):
    copied = _copy_scenario(monkeypatch, tmp_path, ["/copy me"])
    assert copied == ["how did revenue do?"]


def test_copy_all_takes_the_whole_conversation(monkeypatch, tmp_path):
    copied = _copy_scenario(monkeypatch, tmp_path, ["/copy all"])
    assert len(copied) == 1
    assert "how did revenue do?" in copied[0]
    assert "ANSWER" in copied[0]


def test_copy_with_nothing_to_copy_says_so(monkeypatch, tmp_path):
    """Before the first turn the transcript is empty; that is not an error."""
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)
    copied = []

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
            screen.query_one("#chat-input", ChatInput).value = "/copy"
            await pilot.press("enter")
            await pilot.pause()
            assert copied == []
            assert "nothing to copy yet" in _log_text(screen)
    _run(scenario())


def test_copy_with_a_bad_argument_explains_itself(monkeypatch, tmp_path):
    copied = _copy_scenario(monkeypatch, tmp_path, ["/copy sideways"])
    assert copied == []
    assert "takes nothing, 'me' or 'all'" in _copy_scenario.log


# /thinking: reasoning models narrate before answering

THINKY = ("<think>\nThe user asked about revenue. Let me check the context.\n"
          "</think>\n\nRevenue grew 12% in Q3.")


class ThinkingLLM:
    def chat(self, messages, tools=None, on_token=None):
        if on_token is not None:
            on_token(THINKY)
        return {"finish_reason": "stop", "content": THINKY, "tool_calls": None}


def _thinking_components(config_path, model_path, max_tokens=256):
    return WorkflowConfig(model={"name": "m"}), FakeRetriever(), ThinkingLLM()


def test_strip_thinking_removes_the_block_not_the_answer():
    assert chat_screen.strip_thinking(THINKY) == "Revenue grew 12% in Q3."
    # <thinking> is the other spelling, and case must not matter.
    assert chat_screen.strip_thinking("<Thinking>hm</THINKING> hi") == "hi"
    # No block at all: the answer is returned untouched.
    assert chat_screen.strip_thinking("just an answer") == "just an answer"


def test_strip_thinking_drops_an_unclosed_block():
    """Generation stopped mid-thought (Esc, or the token cap ran out).

    Keeping the fragment would leave a dangling "<think>" and a wall of
    reasoning presented as if it were the answer.
    """
    assert chat_screen.strip_thinking("answer <think>still going...") == "answer"
    assert chat_screen.strip_thinking("<think>only reasoning") == ""


def _thinking_turn(monkeypatch, tmp_path, commands=()):
    monkeypatch.setattr(chat_screen, "_build_components", _thinking_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            inp = screen.query_one("#chat-input", ChatInput)
            inp.value = "how did revenue do?"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            for command in commands:
                inp.value = command
                await pilot.press("enter")
                await pilot.pause()
            _thinking_turn.screen = screen
            _thinking_turn.log = _log_text(screen)
    _run(scenario())
    return _thinking_turn


def test_reasoning_is_hidden_by_default_but_kept_in_the_session(monkeypatch, tmp_path):
    result = _thinking_turn(monkeypatch, tmp_path)
    assert "Revenue grew 12% in Q3." in result.log
    assert "Let me check the context" not in result.log     # hidden on screen
    # ...but the raw text is still in the session, so /save keeps everything
    # and /thinking on can bring it back.
    assert "<think>" in result.screen._session.messages[-1]["content"]


def test_thinking_on_redraws_the_transcript_to_show_it(monkeypatch, tmp_path):
    """A toggle that only changed FUTURE answers would leave the reasoning
    you just hid sitting on an append-only log."""
    result = _thinking_turn(monkeypatch, tmp_path, ["/thinking on"])
    assert "Let me check the context" in result.log
    assert "reasoning is now shown" in result.log


def test_bare_thinking_flips_it_back(monkeypatch, tmp_path):
    result = _thinking_turn(monkeypatch, tmp_path, ["/thinking on", "/thinking"])
    assert "reasoning is now hidden" in result.log
    assert "Let me check the context" not in result.log      # redrawn without it


def test_thinking_rejects_nonsense(monkeypatch, tmp_path):
    result = _thinking_turn(monkeypatch, tmp_path, ["/thinking maybe"])
    assert "takes on, off, or nothing" in result.log


def test_copy_follows_what_is_on_screen(monkeypatch, tmp_path):
    """Copy the answer you can see; /thinking on then /copy gets the rest."""
    monkeypatch.setattr(chat_screen, "_build_components", _thinking_components)
    copied = []

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
            inp = screen.query_one("#chat-input", ChatInput)
            inp.value = "how did revenue do?"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            inp.value = "/copy"
            await pilot.press("enter")
            await pilot.pause()
            assert copied[-1] == "Revenue grew 12% in Q3."     # no reasoning

            inp.value = "/thinking on"
            await pilot.press("enter")
            await pilot.pause()
            inp.value = "/copy"
            await pilot.press("enter")
            await pilot.pause()
            assert "<think>" in copied[-1]                     # now it is there
    _run(scenario())


def test_an_answer_that_is_only_reasoning_says_which_knobs_help(monkeypatch, tmp_path):
    """The token cap ran out before the model finished thinking."""
    class OnlyThinking:
        def chat(self, messages, tools=None, on_token=None):
            return {"finish_reason": "stop", "content": "<think>still going",
                    "tool_calls": None}

    monkeypatch.setattr(
        chat_screen, "_build_components",
        lambda c, m, max_tokens=256: (WorkflowConfig(model={"name": "m"}),
                                      FakeRetriever(), OnlyThinking()))

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            screen.query_one("#chat-input", ChatInput).value = "hi"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            text = _log_text(screen)
            assert "only reasoning came back" in text
            assert "/tokens" in text                 # names the fix
    _run(scenario())


# The chat slash menu

def test_typing_slash_opens_a_filtered_chat_menu(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            palette = screen.query_one("#chat-palette", OptionList)
            assert palette.display is False

            screen.query_one("#chat-input", ChatInput).value = "/"
            for _ in range(10):
                await pilot.pause()
                if palette.display and palette.option_count == len(chat_screen.CHAT_COMMANDS):
                    break
            assert palette.display is True
            assert palette.option_count == len(chat_screen.CHAT_COMMANDS)

            screen.query_one("#chat-input", ChatInput).value = "/t"
            for _ in range(10):
                await pilot.pause()
                if palette.option_count == 2:
                    break
            assert palette.option_count == 2          # /thinking and /tokens
    _run(scenario())


def test_choosing_from_the_chat_menu_fills_the_line_for_an_argument(
        monkeypatch, tmp_path):
    """/tokens and /save need a value, so selection inserts, not executes."""
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            inp = screen.query_one("#chat-input", ChatInput)
            palette = screen.query_one("#chat-palette", OptionList)

            inp.value = "/tok"
            for _ in range(10):
                await pilot.pause()
                if palette.display:
                    break
            await pilot.press("down")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert inp.value == "/tokens "            # ready for the number
            assert palette.display is False
    _run(scenario())


def test_escape_closes_the_chat_menu_before_leaving(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            palette = screen.query_one("#chat-palette", OptionList)
            screen.query_one("#chat-input", ChatInput).value = "/"
            for _ in range(10):
                await pilot.pause()
                if palette.display:
                    break
            assert palette.display is True

            await pilot.press("escape")
            await pilot.pause()
            assert palette.display is False
            assert app.screen is screen               # still in chat

            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is not screen
    _run(scenario())


def test_split_thinking_separates_reasoning_from_the_answer():
    """They render differently: markdown eats <think> as an HTML tag."""
    reasoning, answer = chat_screen.split_thinking(THINKY)
    assert "Let me check the context" in reasoning
    assert answer == "Revenue grew 12% in Q3."
    # No narration at all: nothing to show separately.
    assert chat_screen.split_thinking("plain") == ("", "plain")
    # Unclosed: the tail is reasoning, and the answer before it survives.
    assert chat_screen.split_thinking("hi <think>cut off") == ("<think>cut off", "hi")


def test_markdown_in_an_answer_is_rendered_not_printed(monkeypatch, tmp_path):
    """The AI PC transcript was full of literal ### and | table | rows."""
    class MarkdownLLM:
        def chat(self, messages, tools=None, on_token=None):
            return {"finish_reason": "stop", "tool_calls": None,
                    "content": "## Heading\n\nRevenue grew **12%** in Q3."}

    monkeypatch.setattr(
        chat_screen, "_build_components",
        lambda c, m, max_tokens=256: (WorkflowConfig(model={"name": "m"}),
                                      FakeRetriever(), MarkdownLLM()))

    async def scenario():
        app = OvatTUI()
        async with app.run_test(size=(80, 30)) as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            screen.query_one("#chat-input", ChatInput).value = "hi"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            # The answer lives in a Response, which IS a Markdown widget, so
            # Textual renders the heading and the bold. Asserting the widget
            # type is the honest check now: asserting on characters would be
            # testing Textual's markdown parser, not OVAT.
            from ovat.cli.chat_screen import Response
            answers = list(screen.query(Response))
            assert len(answers) == 1
            assert answers[0].text == "## Heading\n\nRevenue grew **12%** in Q3."
    _run(scenario())


# The widget transcript: one Prompt and one Response per turn

def test_a_turn_produces_a_prompt_and_a_response_widget(monkeypatch, tmp_path):
    from ovat.cli.chat_screen import Prompt, Response

    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            screen.query_one("#chat-input", ChatInput).value = "how did revenue do?"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            prompts = list(screen.query(Prompt))
            answers = list(screen.query(Response))
            assert [p.text for p in prompts] == ["how did revenue do?"]
            assert [a.text for a in answers] == ["ANSWER"]
    _run(scenario())


def test_streaming_updates_the_response_widget_in_place(monkeypatch, tmp_path):
    """No separate stream line any more: the answer grows where it will stay.

    That is what retired the tail-clipping and the max-height guard, which
    only existed because the old live line could grow until it pushed the
    input off the bottom of the screen.
    """
    from ovat.cli.chat_screen import Response

    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            assert not screen.query("#chat-stream")     # the old widget is gone
            screen.query_one("#chat-input", ChatInput).value = "hi"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            # One widget held the answer the whole way through.
            assert len(list(screen.query(Response))) == 1
    _run(scenario())


def test_the_transcript_scrolls_rather_than_squeezing_the_input(monkeypatch, tmp_path):
    """The regression the old layout had: a long answer stole the screen."""
    from ovat.cli.chat_screen import Response

    class LongLLM:
        def chat(self, messages, tools=None, on_token=None):
            return {"finish_reason": "stop", "tool_calls": None,
                    "content": "word " * 800}

    monkeypatch.setattr(
        chat_screen, "_build_components",
        lambda c, m, max_tokens=256: (WorkflowConfig(model={"name": "m"}),
                                      FakeRetriever(), LongLLM()))

    async def scenario():
        app = OvatTUI()
        async with app.run_test(size=(100, 30)) as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            inp = screen.query_one("#chat-input", ChatInput)
            inp.value = "hi"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.pause()
            assert len(list(screen.query(Response))) == 1
            # The input is still on screen; the scroll view absorbed the answer.
            assert inp.region.y + inp.size.height <= app.size.height
    _run(scenario())


# Links in an answer come from the MODEL, so a click must not follow them

LINKY = "See [ONNX](https://onnx.ai) for the format."


def test_a_clicked_link_is_copied_not_opened(monkeypatch, tmp_path):
    """Textual's default opens the href in a browser on click.

    These widgets render model output, so one click would launch a browser at
    whatever URL the model wrote. On the AI PC it took the whole TUI down.
    """
    from textual.widgets import Markdown
    from ovat.cli.chat_screen import Response

    class LinkLLM:
        def chat(self, messages, tools=None, on_token=None):
            return {"finish_reason": "stop", "content": LINKY, "tool_calls": None}

    monkeypatch.setattr(
        chat_screen, "_build_components",
        lambda c, m, max_tokens=256: (WorkflowConfig(model={"name": "m"}),
                                      FakeRetriever(), LinkLLM()))
    opened, copied = [], []

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            monkeypatch.setattr(app, "open_url",
                                lambda *a, **k: opened.append(a))
            monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
            screen.query_one("#chat-input", ChatInput).value = "hi"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            answer = screen.query_one(Response)
            assert answer._open_links is False        # the default is disabled
            answer.post_message(Markdown.LinkClicked(answer, "https://onnx.ai"))
            await pilot.pause()
            await pilot.pause()

            assert opened == []                       # no browser was launched
            assert copied == ["https://onnx.ai"]      # the URL is one paste away
            assert "link copied" in _log_text(screen)
            assert app.is_running is True             # and the app survived
    _run(scenario())


# /engine: local openvino_genai, or the full OVMS agent with tools

class FakeAgent:
    """Stands in for what factory.build_agent returns."""

    tools = {"search_docs": {}, "transcribe": {}}
    last_trace = {"turns": [
        {"tool_calls": [{"name": "search_docs"}, {"name": "search_docs"}]},
        {"tool_calls": [{"name": "transcribe"}]},
    ]}

    def run(self, question):
        return "TOOLED ANSWER"


def _engine_screen(monkeypatch, tmp_path, ovms_ok=True):
    """A chat screen whose OVMS engine is faked, local engine unchanged."""
    from ovat.cli.chat_screen import OVMSEngine

    def fake_build_engine(config_path, model_path, engine="local",
                          max_tokens=256):
        if engine == "ovms":
            if not ovms_ok:
                raise RuntimeError("connection refused")
            return OVMSEngine(WorkflowConfig(model={"name": "Qwen3-8B"}),
                              FakeAgent())
        return chat_screen.LocalEngine(*_fake_components(config_path,
                                                         model_path))
    monkeypatch.setattr(chat_screen, "_build_engine", fake_build_engine)


def test_engine_defaults_to_local_and_says_it_has_no_tools(monkeypatch, tmp_path):
    _engine_screen(monkeypatch, tmp_path)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            assert screen._engine_name == "local"
            assert screen._engine.name == "local"
            text = _log_text(screen)
            assert "retrieval only, no tools" in text
    _run(scenario())


def test_engine_ovms_switches_to_the_tool_calling_agent(monkeypatch, tmp_path):
    """This is the path where the workflow's model: section finally matters."""
    _engine_screen(monkeypatch, tmp_path)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            await _send(screen, pilot, "/engine ovms")
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert screen._engine.name == "ovms"
            text = _log_text(screen)
            assert "tools: search_docs, transcribe" in text
            assert "Qwen3-8B" in text
    _run(scenario())


def test_the_ovms_engine_answers_and_reports_the_tools_it_used(monkeypatch, tmp_path):
    _engine_screen(monkeypatch, tmp_path)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            await _send(screen, pilot, "/engine ovms")
            await app.workers.wait_for_complete()
            await pilot.pause()

            await _send(screen, pilot, "what is in my docs?")
            await app.workers.wait_for_complete()
            await pilot.pause()

            text = _log_text(screen)
            assert "TOOLED ANSWER" in text
            # De-duplicated, order preserved: search_docs was called twice.
            assert "tools used: search_docs, transcribe" in text
    _run(scenario())


def test_engine_reports_a_bad_name_and_a_failed_build(monkeypatch, tmp_path):
    _engine_screen(monkeypatch, tmp_path, ovms_ok=False)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)

            await _send(screen, pilot, "/engine sideways")
            await pilot.pause()
            assert "takes local or ovms" in _log_text(screen)
            assert screen._engine_name == "local"      # unchanged

            await _send(screen, pilot, "/engine ovms")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "connection refused" in _log_text(screen)
            assert app.screen is screen                # survived the failure
    _run(scenario())


def test_bare_engine_reports_which_one_is_answering(monkeypatch, tmp_path):
    _engine_screen(monkeypatch, tmp_path)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            await _send(screen, pilot, "/engine")
            await pilot.pause()
            text = _log_text(screen)
            assert "engine: local" in text
            assert "/engine ovms" in text              # and how to change it
    _run(scenario())


# Up/Down history and Ctrl-V paste in the chat input

def test_up_and_down_recall_what_you_asked(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            inp = screen.query_one("#chat-input", ChatInput)
            for question in ("first question", "second question"):
                inp.value = question
                await pilot.press("enter")
                await app.workers.wait_for_complete()
                await pilot.pause()

            inp.value = "half typed"
            await pilot.press("up")
            await pilot.pause()
            assert inp.value == "second question"
            await pilot.press("up")
            await pilot.pause()
            assert inp.value == "first question"
            await pilot.press("down")
            await pilot.pause()
            assert inp.value == "second question"
            await pilot.press("down")
            await pilot.pause()
            assert inp.value == "half typed"      # the draft came back
    _run(scenario())


def test_down_still_steps_into_the_slash_menu(monkeypatch, tmp_path):
    """History must not steal the key the menu already uses."""
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            inp = screen.query_one("#chat-input", ChatInput)
            palette = screen.query_one("#chat-palette", OptionList)
            inp.value = "asked something"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            inp.value = "/"
            await pilot.pause()
            assert palette.display is True
            await pilot.press("down")
            await pilot.pause()
            assert screen.focused is palette       # the menu, not the history
            assert inp.value == "/"
    _run(scenario())






# /load opens a picker of every saved conversation

def _saved(tmp_path, name, questions):
    """Write a session file directly, as /save would."""
    folder = tmp_path / ".ovat" / "sessions"
    folder.mkdir(parents=True, exist_ok=True)
    messages = []
    for question in questions:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": "an answer"})
    (folder / f"{name}.json").write_text(json.dumps(messages), encoding="utf-8")


def test_list_sessions_reports_what_each_one_is(tmp_path):
    """A filename alone does not tell you which conversation this was."""
    _saved(tmp_path, "alpha", ["what is openvino?"])
    _saved(tmp_path, "beta", ["first question", "second question"])

    sessions = chat_screen.list_sessions(str(tmp_path))
    by_name = {s["name"]: s for s in sessions}
    assert by_name["alpha"]["turns"] == 1
    assert by_name["beta"]["turns"] == 2
    assert by_name["alpha"]["first"] == "what is openvino?"
    # Newest first, so the one you just saved is at the top.
    assert [s["name"] for s in sessions] == sorted(
        [s["name"] for s in sessions],
        key=lambda n: by_name[n]["modified"], reverse=True)


def test_a_corrupt_session_does_not_hide_the_others(tmp_path):
    _saved(tmp_path, "good", ["fine"])
    (tmp_path / ".ovat" / "sessions" / "broken.json").write_text(
        "{not json", encoding="utf-8")
    names = [s["name"] for s in chat_screen.list_sessions(str(tmp_path))]
    assert "good" in names and "broken" in names


def test_list_sessions_is_empty_before_anything_is_saved(tmp_path):
    assert chat_screen.list_sessions(str(tmp_path)) == []


def test_bare_load_opens_the_picker_and_loads_the_choice(monkeypatch, tmp_path):
    from ovat.cli.chat_screen import SessionPicker

    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)
    _saved(tmp_path, "demo", ["remember this question"])

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            await _send(screen, pilot, "/load")
            await pilot.pause()
            assert isinstance(app.screen, SessionPicker)

            await pilot.press("enter")          # take the highlighted one
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.screen is screen         # the modal closed
            assert "remember this question" in _log_text(screen)
    _run(scenario())


def test_escaping_the_picker_changes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)
    _saved(tmp_path, "demo", ["not loaded"])

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            await _send(screen, pilot, "/load")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen
            assert "not loaded" not in _log_text(screen)
    _run(scenario())


def test_load_with_a_name_still_skips_the_picker(monkeypatch, tmp_path):
    from ovat.cli.chat_screen import SessionPicker

    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)
    _saved(tmp_path, "demo", ["direct load"])

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            await _send(screen, pilot, "/load demo")
            await pilot.pause()
            assert not isinstance(app.screen, SessionPicker)
            assert "direct load" in _log_text(screen)
    _run(scenario())


def test_load_with_nothing_saved_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            await _send(screen, pilot, "/load")
            await pilot.pause()
            assert "no saved conversations yet" in _log_text(screen)
    _run(scenario())


# The thinking pulse

def test_the_thinking_pulse_appears_while_generating_and_leaves_after(
        monkeypatch, tmp_path):
    """Static text says the app is alive; a moving one says it still is,
    which is the question you have thirty seconds into a CPU generation."""
    from ovat.cli.chat_screen import Thinking

    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)
    seen = []

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            assert not screen.query(Thinking)      # nothing before a question

            original = screen._commit_turn

            def spy(*args, **kwargs):
                seen.append(len(list(screen.query(Thinking))))
                return original(*args, **kwargs)
            monkeypatch.setattr(screen, "_commit_turn", spy)

            screen.query_one("#chat-input", ChatInput).value = "hi"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert seen == [1]                     # it was up during the answer
            assert not screen.query(Thinking)      # and gone once it landed
    _run(scenario())


def test_the_pulse_is_removed_when_the_answer_errors(monkeypatch, tmp_path):
    """Otherwise a failed turn leaves it breathing forever."""
    from ovat.cli.chat_screen import Thinking

    class BrokenLLM:
        def chat(self, messages, tools=None, on_token=None):
            raise RuntimeError("model exploded")

    monkeypatch.setattr(
        chat_screen, "_build_components",
        lambda c, m, max_tokens=1024: (WorkflowConfig(model={"name": "m"}),
                                       FakeRetriever(), BrokenLLM()))

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            screen.query_one("#chat-input", ChatInput).value = "hi"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not screen.query(Thinking)
            assert app.is_running is True
    _run(scenario())


def test_the_pulse_stops_when_the_widget_is_removed(monkeypatch, tmp_path):
    """set_interval is tied to the widget, so removing it stops the timer.

    The previous design rescheduled itself from an animation callback and
    had to be told to check is_mounted; a Textual timer needs no such guard,
    which is most of why it replaced it.
    """
    from ovat.cli.chat_screen import Thinking

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            monkeypatch.setattr(chat_screen, "_build_components",
                                _fake_components)
            screen = await _push_ready_screen(app, pilot, tmp_path)
            pulse = Thinking("● thinking…")
            await screen._view.mount(pulse)
            await pilot.pause()
            assert pulse._timers                      # a timer is running
            await pulse.remove()
            await pilot.pause()
            assert pulse not in screen._view.children  # gone from the tree,
            #                                           so its timer goes too
    _run(scenario())



def test_reasoning_is_folded_away_rather_than_dumped(monkeypatch, tmp_path):
    """/thinking on used to solve "I cannot see it" by creating "I cannot
    see past it". It is a Collapsible now: present, titled, and shut."""
    from textual.widgets import Collapsible
    from ovat.cli.chat_screen import Reasoning

    monkeypatch.setattr(chat_screen, "_build_components", _thinking_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            screen.query_one("#chat-input", ChatInput).value = "hi"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            await _send(screen, pilot, "/thinking on")
            await pilot.pause()

            blocks = list(screen.query(Reasoning))
            assert len(blocks) == 1
            assert isinstance(blocks[0], Collapsible)
            assert blocks[0].collapsed is True          # out of the way
            assert "reasoning" in blocks[0].title       # but labelled
            assert "Let me check the context" in blocks[0].text
    _run(scenario())


# Multi-line prompts, and conversations that accumulate

def test_shift_enter_makes_a_new_line_and_enter_sends(monkeypatch, tmp_path):
    """A chat box that cannot send on Enter is not a chat box; one that
    cannot make a new line cannot compose a long prompt."""
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            inp = screen.query_one("#chat-input", ChatInput)
            inp.focus()
            inp.value = "first line"
            await pilot.press("shift+enter")
            inp.insert("second line")
            await pilot.pause()
            assert "\n" in inp.value                # still composing

            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert inp.value == ""                  # sent and cleared
            assert "first line" in _log_text(screen)
    _run(scenario())


def test_arrows_edit_a_multi_line_draft_instead_of_recalling(monkeypatch, tmp_path):
    """History must not steal the keys that move the cursor once there is
    more than one line to move around in."""
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            inp = screen.query_one("#chat-input", ChatInput)
            inp.focus()
            inp.value = "asked before"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            inp.value = "line one\nline two"        # a multi-line draft
            await pilot.press("up")
            await pilot.pause()
            assert inp.value == "line one\nline two"   # untouched by history
    _run(scenario())


def test_each_conversation_gets_its_own_autosave(monkeypatch, tmp_path):
    """Every chat wrote to "last", so the picker could only ever offer the
    newest and every earlier conversation was silently gone."""
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            first = await _push_ready_screen(app, pilot, tmp_path)
            first.query_one("#chat-input", ChatInput).value = "conversation one"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            app.pop_screen()

            second = ChatScreen("w.yml", "m", cwd=str(tmp_path))
            second._autosave = "auto-different"      # a later second
            app.push_screen(second)
            await app.workers.wait_for_complete()
            await pilot.pause()
            second.query_one("#chat-input", ChatInput).value = "conversation two"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            names = {s["name"] for s in chat_screen.list_sessions(str(tmp_path))}
            assert len(names) == 2                   # both survived
    _run(scenario())


def test_loading_an_old_conversation_keeps_writing_to_it(monkeypatch, tmp_path):
    """Adding to an old chat should extend it, not fork a new autosave."""
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)
    _saved(tmp_path, "yesterday", ["an older question"])

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            await _send(screen, pilot, "/load yesterday")
            await pilot.pause()
            assert screen._autosave == "yesterday"

            screen.query_one("#chat-input", ChatInput).value = "a new question"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            by_name = {s["name"]: s for s
                       in chat_screen.list_sessions(str(tmp_path))}
            assert by_name["yesterday"]["turns"] == 2    # extended, not forked
            assert not [n for n in by_name if n.startswith("auto-")]
    _run(scenario())




def test_the_thinking_indicator_actually_changes_colour():
    """It sat at one colour: styles.animate with an on_complete chain never
    fired at mount. Six samples over two seconds were identical."""
    import asyncio as _asyncio
    from textual.app import App, ComposeResult
    from ovat.cli.chat_screen import Thinking

    class Solo(App):
        def compose(self) -> ComposeResult:
            yield Thinking("● thinking…")

    async def scenario():
        app = Solo()
        async with app.run_test() as pilot:
            widget = app.query_one(Thinking)
            seen = []
            for _ in range(5):
                await _asyncio.sleep(0.25)
                await pilot.pause()
                seen.append(str(widget.styles.color))
            assert len(set(seen)) >= 3             # genuinely moving
    _run(scenario())


def test_the_chat_input_keeps_its_border():
    """compact=True removed the border outright, so the prompt box vanished
    and the placeholder floated loose at the bottom of the screen."""
    from ovat.cli.widgets import ChatInput
    box = ChatInput()
    assert "compact" not in box.classes


def test_loading_a_long_conversation_stays_bounded(monkeypatch, tmp_path):
    """A Markdown widget builds a child per block, row and list item, so a
    long history rebuilt them all and froze the app: measured, thirty turns
    made 3,158 widgets and blocked the UI thread for 2.8s. Only the newest
    answers stay rich; the rest become one widget each."""
    from ovat.cli.chat_screen import PlainAnswer, Response

    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)
    body = "## Heading\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n- one\n- two"
    messages = []
    for i in range(25):
        messages.append({"role": "user", "content": f"question {i}"})
        messages.append({"role": "assistant", "content": body})
    folder = tmp_path / ".ovat" / "sessions"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "long.json").write_text(json.dumps(messages), encoding="utf-8")

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            await screen._load_named("long")
            await pilot.pause()

            rich = list(screen.query(Response))
            plain = list(screen.query(PlainAnswer))
            assert len(rich) == chat_screen.REPLAY_RICH_TURNS   # newest only
            assert len(plain) == 25 - chat_screen.REPLAY_RICH_TURNS
            # Nothing was dropped: every answer is still on screen somewhere.
            assert len(rich) + len(plain) == 25
            assert "Heading" in plain[0].text
    _run(scenario())


def test_rebuilding_the_transcript_mid_answer_does_not_crash(monkeypatch, tmp_path):
    """The crash you hit: /thinking while generating called _replay, which
    removed the Response being streamed into, and committing the answer then
    mounted relative to an orphan.

    MountError: Unable to find relative location of Response() because it has
    no parent -- and the whole TUI went down with it.
    """
    from ovat.cli.chat_screen import Response

    monkeypatch.setattr(chat_screen, "_build_components", _thinking_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            screen.query_one("#chat-input", ChatInput).value = "hi"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            # Orphan the answer widget the way a mid-flight _replay would,
            # then commit into it exactly as the worker does.
            orphan = Response("partial")
            assert orphan.parent is None
            screen._commit_turn(orphan, THINKY, ["finance.md"])
            await pilot.pause()

            assert app.is_running is True                  # no MountError
            text = _log_text(screen)
            assert "Revenue grew 12% in Q3." in text       # answer landed
            assert "finance.md" in text                    # sources too
    _run(scenario())


def test_thinking_and_load_stand_down_while_an_answer_is_running(
        monkeypatch, tmp_path):
    """Both rebuild the transcript, which would delete the widget the answer
    is streaming into."""
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)
    _saved(tmp_path, "demo", ["something"])

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            screen._busy = True                            # mid-generation

            await screen._toggle_thinking("on")
            await pilot.pause()
            assert "still answering" in _log_text(screen)

            await screen._load_named("demo")
            await pilot.pause()
            assert "Esc stops it before loading" in _log_text(screen)
    _run(scenario())


# A reshaping answer must not be written through two paths at once

class SplitTagLLM:
    """Emits a <think> block with the TAGS SPLIT ACROSS TOKENS.

    That split is the whole point and is what a real tokenizer does: "<th"
    arrives first and cannot be recognised as a tag yet, so it is displayed.
    When "ink>" completes the tag the block is stripped and the visible
    answer SHRINKS, which is the one case append-only streaming cannot
    express. Emitting "<think>" whole never reproduces it.
    """

    TOKENS = ("<th", "ink>", "weighing it up", "</th", "ink>",
              "the answer", " continues")

    def chat(self, messages, tools=None, on_token=None):
        for token in self.TOKENS:
            if on_token and on_token(token):
                break
        return {"finish_reason": "stop", "tool_calls": None,
                "content": "<think>weighing it up</think>the answer continues"}


def _record_stream_traffic(monkeypatch):
    """Record write/stop on the stream and update on the answer widget, in
    one ordered list, so their INTERLEAVING can be asserted on."""
    from textual.widgets import Markdown
    from ovat.cli.chat_screen import Response

    calls: list = []
    real_get_stream = Markdown.get_stream
    real_update = Response.update

    class Recorder:
        def __init__(self, inner):
            self._inner = inner

        def write(self, text):
            calls.append(("write", text))
            return self._inner.write(text)

        def stop(self):
            calls.append(("stop", None))
            return self._inner.stop()

    def recording_update(self, markdown=""):
        calls.append(("update", markdown))
        return real_update(self, markdown)

    monkeypatch.setattr(Markdown, "get_stream",
                        lambda widget: Recorder(real_get_stream(widget)))
    monkeypatch.setattr(Response, "update", recording_update)
    return calls


def test_no_write_reaches_the_stream_after_a_direct_update(monkeypatch,
                                                           tmp_path):
    """The desync that broke the answer border.

    MarkdownStream keeps its OWN record of what it has written. update()
    replaces the widget's content without telling it, so any later write()
    appends the next delta onto a STALE buffer: the answer renders over
    itself and the box is drawn to the wrong size. Once the text has had to
    reshape, the stream must be retired for the rest of the turn.
    """
    from ovat.cli.chat_screen import Response

    calls = _record_stream_traffic(monkeypatch)
    monkeypatch.setattr(
        chat_screen, "_build_components",
        lambda c, m, max_tokens=256: (WorkflowConfig(model={"name": "m"}),
                                      FakeRetriever(), SplitTagLLM()))

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            screen.query_one("#chat-input", ChatInput).value = "hi"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            kinds = [kind for kind, _ in calls]
            assert "write" in kinds, "nothing streamed; the fake never ran"
            # Anchor AFTER the first write: the Response is mounted with an
            # empty update before anything streams, and that one is not the
            # reshape being tested.
            first_write = kinds.index("write")
            tail = kinds[first_write:]
            assert "update" in tail, (
                "the answer never reshaped, so this test proves nothing: "
                f"{calls}")
            first_update = first_write + tail.index("update")
            assert "write" not in kinds[first_update:], (
                "wrote to the stream after replacing the widget's content, "
                f"which is the desync that broke the border: {calls}")
            # The answer still lands, in one widget, undamaged.
            answers = list(screen.query(Response))
            assert len(answers) == 1
    _run(scenario())


def test_the_stream_is_only_stopped_once(monkeypatch, tmp_path):
    """The reshape path stops it early; the finish path must not stop it
    again, or a second stop lands on a task that is already unwound."""
    from textual.widgets import Markdown

    stops: list = []
    real_get_stream = Markdown.get_stream

    class Recorder:
        def __init__(self, inner):
            self._inner = inner

        def write(self, text):
            return self._inner.write(text)

        def stop(self):
            stops.append(1)
            return self._inner.stop()

    monkeypatch.setattr(Markdown, "get_stream",
                        lambda widget: Recorder(real_get_stream(widget)))
    monkeypatch.setattr(
        chat_screen, "_build_components",
        lambda c, m, max_tokens=256: (WorkflowConfig(model={"name": "m"}),
                                      FakeRetriever(), SplitTagLLM()))

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            screen.query_one("#chat-input", ChatInput).value = "hi"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(stops) == 1, f"stopped {len(stops)} times"
    _run(scenario())


# Turn separators: a rule between one exchange and the next

def _rules(screen):
    from ovat.cli.chat_screen import TurnRule
    return list(screen.query(TurnRule))


def test_no_rule_appears_before_the_first_question(monkeypatch, tmp_path):
    """A rule at the very top would be a line drawn under nothing."""
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            screen.query_one("#chat-input", ChatInput).value = "first"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert _rules(screen) == []
    _run(scenario())


def test_a_rule_separates_each_turn_from_the_next(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            for question in ("one", "two", "three"):
                screen.query_one("#chat-input", ChatInput).value = question
                await pilot.press("enter")
                await app.workers.wait_for_complete()
                await pilot.pause()
            # Three turns, two gaps between them.
            assert len(_rules(screen)) == 2
    _run(scenario())


def test_rebuilding_the_transcript_does_not_double_the_rules(monkeypatch,
                                                              tmp_path):
    """/thinking rebuilds the whole transcript from the session. If the live
    path and the replay path disagreed, toggling it would either lose the
    separators or stack a second one on every turn."""
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            for question in ("one", "two", "three"):
                screen.query_one("#chat-input", ChatInput).value = question
                await pilot.press("enter")
                await app.workers.wait_for_complete()
                await pilot.pause()
            before = len(_rules(screen))

            screen.query_one("#chat-input", ChatInput).value = "/thinking on"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(_rules(screen)) == before, "the rebuild changed the count"
    _run(scenario())


def test_the_rule_costs_one_row_not_three(monkeypatch, tmp_path):
    """Textual's Rule ships margin: 1 0. Left alone that is three rows per
    turn on top of Prompt's own margin, which on a small terminal is most of
    the screen spent on separators."""
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test(size=(80, 30)) as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            for question in ("one", "two"):
                screen.query_one("#chat-input", ChatInput).value = question
                await pilot.press("enter")
                await app.workers.wait_for_complete()
                await pilot.pause()
            rule = _rules(screen)[0]
            assert rule.outer_size.height == 1, (
                f"the separator takes {rule.outer_size.height} rows")
    _run(scenario())


def test_the_input_has_no_hover_tooltip(monkeypatch, tmp_path):
    """A tooltip was tried here and had to come out.

    Textual renders it as an unstyled block floating over whatever is behind
    it, anchored to the pointer. Over the transcript it covered the last
    answer, and its own text wrapped mid-phrase ("Shift-Enter for a / new
    line"), so the thing meant to explain the input obscured the conversation
    instead. The placeholder already carries the essentials and costs nothing.
    """
    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            assert screen.query_one("#chat-input", ChatInput).tooltip is None
    _run(scenario())


# The header must agree with the engine that is actually loaded

def _header_plain(header) -> str:
    """Static keeps its content under different names across Textual
    versions, so read whichever one this build actually has."""
    for attr in ("renderable", "_renderable", "_content", "content"):
        value = getattr(header, attr, None)
        if value is not None:
            return getattr(value, "plain", str(value))
    return str(header.render())


def test_switching_engine_updates_the_header(monkeypatch, tmp_path):
    """The header is the only thing on screen that names the live engine, and
    it was built once at compose time and never touched again. After
    /engine ovms it still read "engine: local" while the transcript said
    OVMS. Two places disagreeing about that is worse than having neither.
    """
    from textual.widgets import Static

    monkeypatch.setattr(chat_screen, "_build_components", _fake_components)

    built = []

    def fake_build_engine(config_path, model_path, engine="local",
                          max_tokens=chat_screen.DEFAULT_MAX_TOKENS):
        built.append(engine)
        return chat_screen.LocalEngine(
            *_fake_components(config_path, model_path, max_tokens))

    monkeypatch.setattr(chat_screen, "_build_engine", fake_build_engine)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            header = screen.query_one("#chat-header", Static)
            assert "engine: local" in _header_plain(header)

            screen.query_one("#chat-input", ChatInput).value = "/engine ovms"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert "engine: ovms" in _header_plain(header), (
                f"header still says {_header_plain(header)!r}")
            assert "ovms" in built, "the switch never reached _build_engine"
    _run(scenario())


def test_the_header_is_built_in_exactly_one_place():
    """The duplication is what let the two copies drift apart."""
    import inspect

    source = inspect.getsource(chat_screen)
    assert source.count('header.append("OVAT chat"') == 1
