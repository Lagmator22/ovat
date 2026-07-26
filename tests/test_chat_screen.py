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

from textual.widgets import Input, OptionList, RichLog, Static

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
            # The speaker label and the answer are separate lines now: the
            # answer is rendered as markdown, and a markdown block cannot be
            # prefixed inline.
            assert "ovat ›" in text
            assert "ANSWER" in text
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


# /tokens: retune the answer cap without a model reload

class _CapLLM(FakeLLM):
    max_new_tokens = 256


def _components_with_cap(config_path, model_path, max_tokens=256):
    cfg = WorkflowConfig(model={"name": "m"})
    return cfg, FakeRetriever(), _CapLLM()


def _send(screen, pilot, text):
    screen.query_one("#chat-input", Input).value = text
    return pilot.press("enter")


def test_tokens_command_retunes_the_live_provider(monkeypatch, tmp_path):
    """The screen loaded with a 256 cap and no way to change it, so long
    answers were simply cut off. /tokens must take effect with no reload."""
    monkeypatch.setattr(chat_screen, "_build_components", _components_with_cap)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            llm = screen._components[2]
            assert llm.max_new_tokens == 256

            await _send(screen, pilot, "/tokens 2048")
            await pilot.pause()
            assert llm.max_new_tokens == 2048          # same object, no reload

            await _send(screen, pilot, "/tokens 0")
            await pilot.pause()
            assert llm.max_new_tokens is None          # 0 means no cap
            assert "no tokens" in _log_text(screen)
    _run(scenario())


def test_tokens_command_refuses_nonsense_without_crashing(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_screen, "_build_components", _components_with_cap)

    async def scenario():
        app = OvatTUI()
        async with app.run_test() as pilot:
            screen = await _push_ready_screen(app, pilot, tmp_path)
            llm = screen._components[2]

            await _send(screen, pilot, "/tokens lots")
            await pilot.pause()
            assert "wants a number" in _log_text(screen)
            assert llm.max_new_tokens == 256           # unchanged

            await _send(screen, pilot, "/tokens -1")
            await pilot.pause()
            assert "cannot be negative" in _log_text(screen)
            assert llm.max_new_tokens == 256
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
            inp = screen.query_one("#chat-input", Input)
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
            screen.query_one("#chat-input", Input).value = "/copy"
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
            inp = screen.query_one("#chat-input", Input)
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
            inp = screen.query_one("#chat-input", Input)
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
            screen.query_one("#chat-input", Input).value = "hi"
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

            screen.query_one("#chat-input", Input).value = "/"
            await pilot.pause()
            assert palette.display is True
            assert palette.option_count == len(chat_screen.CHAT_COMMANDS)

            screen.query_one("#chat-input", Input).value = "/t"
            await pilot.pause()
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
            inp = screen.query_one("#chat-input", Input)
            palette = screen.query_one("#chat-palette", OptionList)

            inp.value = "/tok"
            await pilot.pause()
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
            screen.query_one("#chat-input", Input).value = "/"
            await pilot.pause()
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
            screen.query_one("#chat-input", Input).value = "hi"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            text = _log_text(screen)
            assert "Heading" in text
            assert "12%" in text
            assert "##" not in text        # rendered as a heading...
            assert "**" not in text        # ...and as bold, not asterisks
    _run(scenario())
