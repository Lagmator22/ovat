# tests/test_agent_tui.py
"""The agent TUI (`ovat agent`), driven headless with Textual's Pilot.

The session is a fake that plays a scripted turn on a thread, exactly as
the real AgentSession reports one, so these tests cover the drawing and
the keys, not the agent. Every wait is a bounded poll (the Windows runner
is slower; one fixed pause has failed there before).
"""
import asyncio
import threading
import time

import pytest

pytest.importorskip("textual")

from ovat.agent.events import (SUBAGENT_STATUS, TOOL_ENDED,  # noqa: E402
                               TOOL_STARTED, Event)
from ovat.agent.permissions import ApprovalBroker  # noqa: E402
from ovat.agentic import session as s  # noqa: E402
from ovat.tui.app import OvatAgentApp  # noqa: E402
from ovat.tui.widgets.approval import ApprovalPanel  # noqa: E402
from ovat.tui.widgets.banner import Banner  # noqa: E402
from ovat.tui.widgets.input import CompletionPopup, InputBox  # noqa: E402
from ovat.tui.widgets.messages import (AssistantMessage, Notice,  # noqa: E402
                                      ShellBlock, UserMessage)
from ovat.tui.widgets.tools import ToolGroup  # noqa: E402

INFO = {"workflow": "w.yml", "name": "w", "model": "Qwen3.5-4B-int4-ov",
        "device": "GPU", "provider": "ovms", "engine": "native",
        "tools": [{"name": "search_docs", "kind": "builtin",
                   "permission": "allow"},
                  {"name": "delegate", "kind": "subagent",
                   "permission": "allow"}],
        "subagents": [{"name": "researcher", "description": "reads docs",
                       "where": "workflow"}]}


class FakeSession:
    def __init__(self, workflow, sink, cwd=None):
        self.sink, self.workflow, self.info = sink, workflow, {}
        self.broker = ApprovalBroker(
            on_request=lambda r: sink(s.ApprovalNeeded(r)))
        self.busy = False
        self.serving = False
        self.last_trace, self.last_answer = {}, ""
        self.totals = {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.agent = None
        self.answer = "New hires get **20 days**. See C:\\hr\\[/leave].md"
        self.ask_approval = False
        self.approved = None
        self.hold = threading.Event()
        self.hold.set()
        self.cancelled = False

    def start(self, engine=None):
        self.info = INFO
        self.sink(s.Ready(INFO))

    def ask(self, text):
        self.busy = True

        def go():
            self.sink(s.TurnStarted(text))
            for ev in (
                Event(TOOL_STARTED, {"call_id": "1", "name": "search_docs",
                                     "arguments": {"query": text}}),
                Event(TOOL_ENDED, {"call_id": "1", "name": "search_docs",
                                   "status": "ok", "duration_s": 0.4,
                                   "result_chars": 9}),
                Event(SUBAGENT_STATUS, {"agent": "researcher",
                                        "text": "calling search_docs"}),
            ):
                self.sink(s.AgentActivity(ev))
            if self.ask_approval:
                self.approved = self.broker("subagent cloud", {"task": "t"},
                                            "sends the task away")
            self.hold.wait(10)
            self.busy = False
            self.sink(s.TurnDone(self.answer, 1.5, {
                "tool_calls": 1, "prompt_tokens": 100,
                "completion_tokens": 20, "latency_s": 1.5},
                ["/docs/leave.md"], False, self.cancelled))
        threading.Thread(target=go, daemon=True).start()
        return True

    def ask_subagent(self, name, task):
        return self.ask(task)

    def cancel(self):
        if self.busy:
            self.cancelled = True
            self.hold.set()
            return True
        return False

    def close(self):
        self.hold.set()

    def run_shell(self, command):
        def go():
            self.sink(s.ShellLine("hello [/bold] world"))
            self.sink(s.ShellDone(command, 0, 0.1))
        threading.Thread(target=go, daemon=True).start()
        return True

    def new_conversation(self):
        return None


async def _until(pilot, check, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "condition not met in time"
        await pilot.pause(0.05)


def _app(session_holder=None, hardware=False):
    def factory(workflow, sink, cwd=None):
        fake = FakeSession(workflow, sink, cwd)
        if session_holder is not None:
            session_holder.append(fake)
        return fake
    return OvatAgentApp("w.yml", session_factory=factory, animate=False,
                        history_file=None, hardware=hardware)


async def _type(pilot, text):
    for ch in text:
        await pilot.press("space" if ch == " " else ch)


def run(coro):
    asyncio.run(coro)


def test_the_banner_and_bar_show_the_session():
    async def go():
        app = _app()
        async with app.run_test(size=(110, 30)) as pilot:
            await _until(pilot, lambda: "Qwen3.5" in str(
                app.query_one("#bottom-bar").render()))
            bar = str(app.query_one("#bottom-bar").render())
            assert "GPU" in bar and "native" in bar
            assert app.query_one(Banner)
    run(go())


def test_a_question_becomes_a_folded_tool_line_and_an_answer():
    async def go():
        app = _app()
        async with app.run_test(size=(110, 30)) as pilot:
            await _type(pilot, "How many leave days?")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(AssistantMessage))
            kinds = [type(w).__name__ for w in
                     app.query_one("#messages").children]
            assert kinds == ["UserMessage", "ToolGroup", "AssistantMessage"]
            group = app.query_one(ToolGroup)
            assert str(group._header()).startswith("⏵ Searched docs")
            group.toggle()
            assert "search_docs" in str(group._body())
            assert "researcher: calling search_docs" in str(group._body())
            assert not app.query("LoadingLine")
    run(go())


def test_markup_like_text_in_an_answer_or_output_cannot_break_it():
    async def go():
        app = _app()
        async with app.run_test(size=(110, 30)) as pilot:
            await _type(pilot, "q")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(AssistantMessage))
            await _type(pilot, "!echo x")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(ShellBlock)
                         and "done" in str(app.query_one(ShellBlock)._end.render()))
            assert "[/bold]" in str(app.query_one(ShellBlock)._out.render())
    run(go())


def test_a_question_typed_while_busy_is_given_back():
    async def go():
        holder = []
        app = _app(holder)
        async with app.run_test(size=(110, 30)) as pilot:
            await _until(pilot, lambda: holder)
            holder[0].hold.clear()
            await _type(pilot, "first")
            await pilot.press("enter")
            await _until(pilot, lambda: holder[0].busy)
            await _type(pilot, "second")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(Notice))
            assert app.query_one(InputBox).area.text == "second"
            holder[0].hold.set()
    run(go())


def test_the_approval_panel_replaces_the_input_and_ignores_early_keys():
    async def go():
        holder = []
        app = _app(holder)
        async with app.run_test(size=(110, 30)) as pilot:
            await _until(pilot, lambda: holder)
            holder[0].ask_approval = True
            await _type(pilot, "q")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(ApprovalPanel))
            assert app.query_one(InputBox).display is False
            await pilot.press("1")          # inside the 0.5 s grace: ignored
            assert holder[0].approved is None
            await pilot.pause(0.6)
            await pilot.press("1")
            await _until(pilot, lambda: holder[0].approved is not None)
            assert holder[0].approved is True
            await _until(pilot, lambda: not app.query(ApprovalPanel))
            assert app.query_one(InputBox).display is True
    run(go())


def test_esc_on_the_approval_panel_says_no():
    async def go():
        holder = []
        app = _app(holder)
        async with app.run_test(size=(110, 30)) as pilot:
            await _until(pilot, lambda: holder)
            holder[0].ask_approval = True
            await _type(pilot, "q")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(ApprovalPanel))
            await pilot.pause(0.6)
            await pilot.press("escape")
            await _until(pilot, lambda: holder[0].approved is not None)
            assert holder[0].approved is False
    run(go())


def test_esc_stops_a_running_answer():
    async def go():
        holder = []
        app = _app(holder)
        async with app.run_test(size=(110, 30)) as pilot:
            await _until(pilot, lambda: holder)
            holder[0].hold.clear()
            await _type(pilot, "long")
            await pilot.press("enter")
            await _until(pilot, lambda: holder[0].busy)
            await pilot.press("escape")
            await _until(pilot, lambda: app.query(AssistantMessage))
            assert "stopped" in str(
                app.query_one(AssistantMessage).query_one(".footer").render())
    run(go())


def test_bang_runs_a_shell_command_once():
    async def go():
        app = _app()
        async with app.run_test(size=(110, 30)) as pilot:
            await _type(pilot, "!echo hi")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(ShellBlock))
            assert not app.query(UserMessage)     # the block shows it once
    run(go())


def test_slash_shows_completions_and_enter_runs_help():
    async def go():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            await pilot.press("/")
            popup = app.query_one(CompletionPopup)
            await _until(pilot, lambda: popup.shown)
            assert popup.selected().name == "help"
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(Notice))
            assert "Commands" in str(app.query_one(Notice).render())
    run(go())


def test_an_unknown_command_suggests_a_near_one():
    async def go():
        app = _app()
        async with app.run_test(size=(110, 30)) as pilot:
            await _type(pilot, "/tols x")
            await pilot.press("escape")      # close the popup first
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(Notice))
            text = str(app.query_one(Notice).render())
            assert "No command /tols" in text and "/tools" in text
    run(go())


def test_ctrl_c_twice_quits_when_idle():
    async def go():
        app = _app()
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.press("ctrl+c")
            assert app.is_running
            await pilot.press("ctrl+c")
            await _until(pilot, lambda: not app.is_running)
    run(go())


def test_the_prompt_stays_on_screen_at_80x24():
    async def go():
        app = _app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause(0.2)
            area = app.query_one(InputBox).area
            assert 0 <= area.region.y < 24 and area.region.height >= 1
    run(go())


def test_there_are_no_tooltips():
    async def go():
        app = _app()
        async with app.run_test(size=(110, 30), tooltips=True) as pilot:
            await pilot.pause(0.2)
            assert all(w.tooltip is None for w in app.screen.walk_children())
    run(go())


def test_no_widget_reuses_a_textual_private_name():
    # self._running on a widget silently shadowed Textual's own message-pump
    # flag: the tool group never showed it had finished. Any private name a
    # Textual base class already uses is a bug waiting to happen.
    import re
    from pathlib import Path

    from textual.app import App
    from textual.widget import Widget
    from textual.widgets import TextArea

    taken = set(dir(Widget)) | set(dir(TextArea)) | set(dir(App))
    taken |= set(vars(Widget()))
    root = Path(__file__).parents[1] / "ovat" / "tui"
    for path in root.rglob("*.py"):
        mine = set(re.findall(r"self\.(_[a-z_]+)\s*=",
                              path.read_text(encoding="utf-8")))
        assert not mine & taken, (path, sorted(mine & taken))


def test_the_bar_shows_a_live_hardware_reading_without_inventing_one():
    class Monitor:
        def sample(self):
            return {"cpu": 23.4, "ram": 61.0}     # no NPU on this machine

        def close(self):
            pass

    async def go():
        app = _app(hardware=Monitor())
        async with app.run_test(size=(140, 30)) as pilot:
            await _until(pilot, lambda: "CPU 23%" in str(
                app.query_one("#bottom-bar").render()), timeout=6)
            bar = str(app.query_one("#bottom-bar").render())
            assert "RAM 61%" in bar and "NPU" not in bar
    run(go())
