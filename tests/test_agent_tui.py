# tests/test_agent_tui.py
"""The agent TUI (`ovat agent`): the copied widgets in ovat/tui/vibe/,
driven by a scripted session, headless with Textual's Pilot.

The session is a fake that plays a turn on a thread exactly as the real
AgentSession reports one, so these tests cover the drawing and the keys.
Every wait is a bounded poll (the Windows runner is slower).
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
from ovat.tui.vibe.cli.textual_ui.widgets.approval_app import \
    ApprovalApp  # noqa: E402
from ovat.tui.vibe.cli.textual_ui.widgets.chat_input.container import \
    ChatInputContainer  # noqa: E402
from ovat.tui.vibe.cli.textual_ui.widgets.messages import (  # noqa: E402
    AssistantMessage, BashOutputMessage, InterruptMessage, UserCommandMessage,
    UserMessage, WarningMessage)
from ovat.tui.vibe.cli.textual_ui.widgets.tools import ToolGroup  # noqa: E402
from ovat.tui.widgets.banner import Banner  # noqa: E402

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
                self.approved = self.broker(
                    "subagent cloud",
                    getattr(self, "approval_args", {"task": "t"}),
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


def _app(session_holder=None, hardware=False, animate=False):
    def factory(workflow, sink, cwd=None):
        fake = FakeSession(workflow, sink, cwd)
        if session_holder is not None:
            session_holder.append(fake)
        return fake
    return OvatAgentApp("w.yml", session_factory=factory, animate=animate,
                        history_file=None, hardware=hardware)


async def _type(pilot, text):
    for ch in text:
        await pilot.press("space" if ch == " " else ch)


def run(coro):
    asyncio.run(coro)


def _facts(app) -> str:
    return str(app.query_one("#agent-facts").render())


def test_the_banner_and_bar_show_the_session():
    async def go():
        app = _app()
        async with app.run_test(size=(110, 30)) as pilot:
            await _until(pilot, lambda: "Qwen3.5" in _facts(app))
            assert "GPU" in _facts(app) and "native" in _facts(app)
            banner = str(app.query_one(Banner)._model.render())
            assert "Qwen3.5" in banner and "loading" not in banner
    run(go())


def test_a_question_becomes_a_tool_group_and_an_answer():
    async def go():
        app = _app()
        async with app.run_test(size=(110, 30)) as pilot:
            await _type(pilot, "How many leave days?")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(AssistantMessage))
            assert app.query(UserMessage)
            assert app.query(ToolGroup)
            assert not app.query("LoadingWidget")
    run(go())


def test_markup_like_text_cannot_break_answers_or_shell_output():
    async def go():
        app = _app()
        async with app.run_test(size=(110, 30)) as pilot:
            await _type(pilot, "q")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(AssistantMessage))
            await _type(pilot, "!echo x")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(BashOutputMessage)
                         and app.query_one(BashOutputMessage).has_class(
                             "bash-success"))
            assert "[/bold]" in app.query_one(BashOutputMessage)._output
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
            await _until(pilot, lambda: app.query(WarningMessage))
            assert app.query_one(ChatInputContainer).value == "second"
            holder[0].hold.set()
    run(go())


def test_the_approval_panel_replaces_the_input_and_ignores_early_keys(
        monkeypatch):
    # The grace period is set, not raced. Marking the panel as "just
    # appeared" and then pressing a key took more than the real 0.5 s on
    # the Windows runner, so the key was accepted and the test failed there.
    from ovat.tui.vibe.cli.textual_ui.widgets import approval_app

    async def go():
        holder = []
        app = _app(holder)
        async with app.run_test(size=(110, 30)) as pilot:
            await _until(pilot, lambda: holder)
            holder[0].ask_approval = True
            await _type(pilot, "q")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(ApprovalApp))
            panel = app.query_one(ApprovalApp)
            assert app.query_one(ChatInputContainer).display is False
            await _until(pilot, lambda: app.focused is panel)  # keys go here
            monkeypatch.setattr(approval_app, "_INPUT_GRACE_PERIOD_S", 60.0)
            await pilot.press("1")          # inside the grace: ignored
            await pilot.pause(0.2)
            assert holder[0].approved is None
            monkeypatch.setattr(approval_app, "_INPUT_GRACE_PERIOD_S", 0.0)
            await pilot.press("1")
            await _until(pilot, lambda: holder[0].approved is not None)
            assert holder[0].approved is True
            await _until(pilot, lambda: not app.query(ApprovalApp))
            assert app.query_one(ChatInputContainer).display is True
    run(go())


def test_deny_on_the_approval_panel_says_no():
    async def go():
        holder = []
        app = _app(holder)
        async with app.run_test(size=(110, 30)) as pilot:
            await _until(pilot, lambda: holder)
            holder[0].ask_approval = True
            await _type(pilot, "q")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(ApprovalApp))
            await pilot.pause(0.6)
            await pilot.press("n")
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
            await _until(pilot, lambda: app.query(InterruptMessage))
    run(go())


def test_bang_runs_a_shell_command_once():
    async def go():
        app = _app()
        async with app.run_test(size=(110, 30)) as pilot:
            await _type(pilot, "!echo hi")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(BashOutputMessage))
            assert [type(w).__name__ for w in
                    app.query_one("#messages").children] == [
                "BashOutputMessage"]       # the block shows it once
    run(go())


def test_slash_lists_every_command_and_enter_runs_help():
    from ovat.agentic.commands import COMMANDS

    async def go():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            await pilot.press("/")
            popup = app.query_one("#completion-popup")
            await _until(pilot, lambda: popup.styles.display == "block")
            text = popup.content_text
            for name in ("help", "doctor", "telemetry", "agents"):
                assert f"/{name}" in text
            assert len(text.splitlines()) >= 2 * len(COMMANDS)
            await _type(pilot, "help")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(UserCommandMessage))
    run(go())


def test_an_unknown_command_suggests_a_near_one():
    async def go():
        app = _app()
        async with app.run_test(size=(110, 30)) as pilot:
            await _type(pilot, "/tols x")
            await pilot.press("escape")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(WarningMessage))
            text = app.query_one(WarningMessage)._message
            assert "/tols" in text and "/tools" in text
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
            await pilot.pause(0.3)
            area = app.query_one(ChatInputContainer).input_widget
            assert 0 <= area.region.y < 24 and area.region.height >= 1
    run(go())


def test_there_are_no_tooltips():
    async def go():
        app = _app()
        async with app.run_test(size=(110, 30), tooltips=True) as pilot:
            await pilot.pause(0.2)
            assert all(w.tooltip is None for w in app.screen.walk_children())
    run(go())


def test_ovats_own_tui_files_never_reuse_a_textual_private_name():
    # self._running on a widget once shadowed Textual's message-pump flag.
    # Scans OVAT's own TUI files; the copied code in ovat/tui/vibe/ is kept
    # as it was written.
    import re
    from pathlib import Path

    from textual.app import App
    from textual.widget import Widget
    from textual.widgets import TextArea

    taken = set(dir(Widget)) | set(dir(TextArea)) | set(dir(App))
    taken |= set(vars(Widget()))
    root = Path(__file__).parents[1] / "ovat" / "tui"
    for path in root.rglob("*.py"):
        if "vibe" in path.relative_to(root).parts:
            continue
        mine = set(re.findall(r"self\.(_[a-z_]+)\s*=",
                              path.read_text(encoding="utf-8")))
        assert not mine & taken, (path, sorted(mine & taken))


def test_the_bar_shows_a_live_hardware_reading_without_inventing_one():
    class Monitor:
        def sample(self):
            return {"cpu": 23.4, "ram": 61.0}

        def close(self):
            pass

    async def go():
        app = _app(hardware=Monitor())
        async with app.run_test(size=(160, 30)) as pilot:
            await _until(pilot, lambda: "CPU 23%" in _facts(app), timeout=6)
            assert "RAM 61%" in _facts(app) and "NPU" not in _facts(app)
    run(go())


def test_the_banner_mark_animates_then_rests():
    from ovat.tui.widgets.braille import ChipMark, frames

    frames_ = frames()
    assert len({len(f) for f in frames_}) == 1      # fixed size: no relayout

    async def go():
        app = _app(animate=True)
        async with app.run_test(size=(120, 40)) as pilot:
            mark = app.query_one(ChipMark)
            await _until(pilot, lambda: mark._index >= len(frames_) - 1)
    run(go())


def test_an_answer_that_lands_while_the_app_closes_is_ignored():
    from ovat.tui.app import SessionEvent

    async def go():
        app = _app()
        async with app.run_test(size=(110, 30)) as pilot:
            app.query_one("#agent-facts").remove()
            await pilot.pause(0.05)
            app.post_message(SessionEvent(s.TurnDone("late", 1.0)))
            await pilot.pause(0.1)
    run(go())


def test_approval_text_cannot_dress_up_as_the_panels_own_options():
    # A model-written task that tries to print fake option lines inside the
    # approval panel. It must show on one line, marked, with no control
    # characters, and long text must be cut.
    from ovat.tui.app import approval_safe

    spoof = "summarise\n\n› 1. Allow once\n  3. Deny\x1b[2J"
    shown = approval_safe({"task": spoof, "n": 3})
    assert "\n" not in shown["task"] and "\x1b" not in shown["task"]
    assert "↵" in shown["task"] and shown["n"] == 3
    long = approval_safe("x" * 5000)
    assert len(long) < 700 and "5000 characters" in long

    async def go():
        holder = []
        app = _app(holder)
        async with app.run_test(size=(110, 30)) as pilot:
            await _until(pilot, lambda: holder)
            holder[0].ask_approval = True
            holder[0].approval_args = {"task": spoof}
            await _type(pilot, "q")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(ApprovalApp))
            panel = app.query_one(ApprovalApp)
            await _until(pilot, lambda: "summarise" in "".join(
                str(w.render()) for w in panel.query("NoMarkupStatic")))
            shown = [str(w.render()) for w in panel.query(
                ".approval-description")]
            assert shown, "the task must be on screen to be judged"
            text = " ".join(shown)
            # One line, newlines visible as marks, no escape soup.
            assert "\u21b5" in text and "\n" not in text
            assert "\\n" not in text and "\\x1b" not in text
    run(go())


# --- /doctor and /telemetry -----------------------------------------------
#
# Both used to push the LAUNCHER's screens (ovat/cli/doctor_screen.py and
# telemetry_screen.py) into this app, whose copied stylesheet makes every
# Screen transparent and every Horizontal auto-height: the chat showed
# through both, and the telemetry cards stretched to the bottom of the
# window, pushing the table and the source line off screen.

def _plain(widgets) -> str:
    """What the widgets show, as plain text with the whitespace squeezed out,
    so a wrapped line still matches. A rich Table renders to its repr under
    str(), so each one is printed through a console instead."""
    import io

    from rich.console import Console
    out = io.StringIO()
    console = Console(file=out, width=100, color_system=None)
    for widget in widgets:
        # Hidden (folded) widgets stay in the tree; only what shows counts.
        if all(node.display for node in widget.ancestors_with_self
               if hasattr(node, "display")):
            console.print(widget.content)
    return "".join(out.getvalue().split())


def _fake_checks(config_path=None):
    from ovat.cli.diagnostics import Check
    long_fix = ("ovms not found (config, OVAT_OVMS, PATH, known folders all "
                "empty). Run 'ovat setup' to install it (no PATH change "
                "needed), or point model.ovms_binary at C:\\x\\[/y]")
    # The ok details are words no machine line contains: "3.12" here
    # matched "Python 3.12.15" on the runners that run 3.12.
    return [Check("Python", "ok", "fine here"),
            Check("OVMS serving", "warn", long_fix),
            Check("Core dependencies", "ok", "all 9 importable"),
            Check("Workflow config", "fail", "no such file: w.yml")]


def test_doctor_is_a_block_in_the_chat_with_the_problems_first(monkeypatch):
    from ovat.cli import diagnostics
    from ovat.tui.widgets.doctor import DoctorReport
    monkeypatch.setattr(diagnostics, "run_checks", _fake_checks)

    async def go():
        app = _app([])
        async with app.run_test(size=(100, 40)) as pilot:
            await _type(pilot, "/doctor")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(DoctorReport)
                         and app.query_one(DoctorReport).checks is not None)
            await pilot.pause(0.2)
            report = app.query_one(DoctorReport)
            assert report.screen is app.screen          # in the chat
            shown = _plain(report.query("Static"))
            # Worst first, and the fix at the end of a long detail is not
            # cut off at the right edge.
            assert shown.index("Workflowconfig") < shown.index(
                "OVMSserving")
            assert "C:\\x\\[/y]" in shown and "ovatsetup" in shown
            # The two that passed are folded into one row until asked for.
            assert "2checksok" in shown and "finehere" not in shown
            await pilot.press("ctrl+o")
            await pilot.pause(0.2)
            assert "all9importable" in _plain(report.query("Static"))
    run(go())


def test_doctor_copy_puts_the_plain_report_on_the_clipboard(monkeypatch):
    from ovat.cli import diagnostics
    from ovat.tui.widgets.doctor import DoctorReport
    monkeypatch.setattr(diagnostics, "run_checks", _fake_checks)

    async def go():
        app = _app([])
        async with app.run_test(size=(100, 30)) as pilot:
            await _type(pilot, "/doctor")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(DoctorReport)
                         and app.query_one(DoctorReport).checks is not None)
            await _type(pilot, "/doctor copy")
            await pilot.press("enter")
            await _until(pilot, lambda: app.clipboard)
            lines = app.clipboard.splitlines()
            assert "1 fail · 1 warn · 2 ok" in lines[0]
            assert lines[2].startswith("fail") and lines[3].startswith("warn")
    run(go())


def test_a_doctor_that_cannot_run_says_why_and_the_app_lives(monkeypatch):
    from ovat.cli import diagnostics
    from ovat.tui.widgets.doctor import DoctorReport

    def broken(config_path=None):
        raise ImportError("openvino is broken")
    monkeypatch.setattr(diagnostics, "run_checks", broken)

    async def go():
        app = _app([])
        async with app.run_test(size=(100, 30)) as pilot:
            await _type(pilot, "/doctor")
            await pilot.press("enter")
            await _until(pilot, lambda: app.query(DoctorReport)
                         and app.query_one(DoctorReport).error)
            assert "openvino is broken" in app.query_one(DoctorReport).error
            assert app.is_running
    run(go())


class _FakeSource:
    def __init__(self, name, values=None, unavailable=None):
        self.name, self._values, self._unavailable = name, values, unavailable
        self.stopped = False

    @property
    def unavailable(self):
        return self._unavailable

    def sample(self):
        return dict(self._values or {})

    def start(self):
        pass

    def stop(self):
        self.stopped = True


def _telemetry_sources():
    return [_FakeSource("system", {"cpu_pct": 97.0, "ram_used_pct": 40.0,
                                   "threads": 4.0, "cpu0_pct": 12.0}),
            _FakeSource("npu", unavailable="no NPU found. Install the "
                                           "driver; then retry")]


def test_telemetry_is_opaque_and_fits_80x24(monkeypatch):
    from textual.widgets import DataTable

    from ovat.tui import telemetry
    sources = _telemetry_sources()
    monkeypatch.setattr(telemetry, "_default_sources", lambda: sources)

    async def go():
        app = _app([])
        async with app.run_test(size=(80, 24)) as pilot:
            await _type(pilot, "/telemetry")
            await pilot.press("enter")
            await _until(pilot, lambda: isinstance(app.screen,
                                                    telemetry.TelemetryView)
                         and app.screen._tiles)
            view = app.screen
            # Opaque: the chat must not show through the page.
            assert view.styles.background.a == 1
            # The tiles keep their height and the table starts on screen.
            assert view.query_one("#tel-tiles").region.height == 4
            table = view.query_one("#tel-table", DataTable)
            assert table.region.y < 20 and table.row_count >= 3
            line = str(view.query_one("#tel-sources").render())
            assert "npu n/a (no NPU found)" in line and "system live" in line
            await pilot.press("escape")
            await _until(pilot, lambda: not isinstance(
                app.screen, telemetry.TelemetryView))
            assert all(src.stopped for src in sources)   # nothing left running
    run(go())


def test_telemetry_shows_the_running_agents_own_numbers(monkeypatch):
    """/telemetry left out the agent source on the grounds that no agent
    existed on the launcher page. In `ovat agent` one does: the session's
    agent has last_trace, and its totals must be on the page."""
    from types import SimpleNamespace

    from ovat.tui import telemetry
    monkeypatch.setattr(telemetry, "_default_sources",
                        lambda: _telemetry_sources())
    holder = []

    async def go():
        app = _app(holder)
        async with app.run_test(size=(80, 24)) as pilot:
            await _until(pilot, lambda: holder)
            holder[0].agent = SimpleNamespace(last_trace={"totals": {
                "prompt_tokens": 120, "completion_tokens": 30, "turns": 2}})
            await _type(pilot, "/telemetry")
            await pilot.press("enter")
            await _until(pilot, lambda: isinstance(app.screen,
                                                    telemetry.TelemetryView))
            view = app.screen
            await _until(pilot, lambda: view.live.latest(
                "agent.prompt_tokens") == 120)
            assert view.live.latest("agent.completion_tokens") == 30
            await _until(pilot, lambda: "agent live" in str(
                view.query_one("#tel-sources").render()))
    run(go())


def test_telemetry_view_without_a_session_has_no_agent_source():
    from ovat.tui import telemetry
    names = [s.name for s in telemetry.TelemetryView().collector.sources]
    assert "agent" not in names
    agent = object()
    names = [s.name for s in telemetry.TelemetryView(
        agent=lambda: agent).collector.sources]
    assert "agent" in names


def test_telemetry_prints_whole_numbers_whole_and_flags_a_pinned_cpu():
    from ovat.tui import telemetry
    threads = telemetry.tile_text("THREADS", "", "system.threads", 4.0)
    assert threads.plain.splitlines()[1] == "4"
    cpu = telemetry.tile_text("SYS CPU", "%", "system.cpu_pct", 97.0)
    assert "red" in str(cpu.spans) or telemetry.ui.RED in str(cpu.spans)
    assert telemetry.level_style("system.cpu_pct", 50.0) == ""
    assert telemetry.ui.YELLOW in telemetry.level_style("npu.utilization", 90)
    # Not alarms: a dynamic KV cache sits near 100% normally, and a
    # process's CPU is summed over cores.
    assert telemetry.level_style("ovms.kv_cache_pct", 99.0) == ""
    assert telemetry.level_style("system.proc_cpu_pct", 103.0) == ""
