# tests/test_vibe_agent_tui.py
"""`ovat agent` on the full copied Vibe app (ovat/tui/agent_app.py), driven
headless with Textual's Pilot.

Nothing in the middle is faked: the app talks JSON-RPC to the REAL OVAT app
server (ovat/tui/server/) through Vibe's own client, and only the AgentSession
at the far end is a fake that plays a turn on a thread, exactly as the real
one reports it (the same fake tests/test_ovat_app_server.py uses). So these
tests cover the app, the wire and the server together; the agent and the
model are the only parts not here.

Every wait is a bounded poll (the Windows runner is slower), and nothing uses
pytest-asyncio: each test is one `asyncio.run`.
"""
import asyncio
import re
import threading
import time
from contextlib import asynccontextmanager
from html import unescape
from pathlib import Path

import pytest

pytest.importorskip("textual")

from ovat.agent.permissions import ALWAYS, ONCE, REFUSE, SESSION  # noqa: E402
from ovat.agent.session import Session  # noqa: E402
from ovat.agentic import commands as ovat_commands  # noqa: E402
from ovat.agentic import session as s  # noqa: E402
from ovat.tui.agent_app import OvatAgentApp  # noqa: E402
from ovat.tui.server import start_ovat_app_server  # noqa: E402
from ovat.tui.vibe.cli import clipboard  # noqa: E402
from ovat.tui.vibe.cli.textual_ui.widgets import approval_app  # noqa: E402
from ovat.tui.vibe.cli.textual_ui.widgets.approval_app import \
    ApprovalApp  # noqa: E402
from ovat.tui.vibe.cli.textual_ui.widgets.chat_input.container import \
    ChatInputContainer  # noqa: E402
from ovat.tui.vibe.cli.textual_ui.widgets.messages import (  # noqa: E402
    AssistantMessage, ErrorMessage, InterruptMessage, ReasoningMessage,
    SlashCommandMessage, UserCommandMessage, UserMessage, WarningMessage)
from ovat.tui.vibe.cli.textual_ui.widgets.theme_picker import \
    ThemePickerApp  # noqa: E402
from ovat.tui.vibe.cli.textual_ui.widgets.tools import ToolGroup  # noqa: E402
from ovat.tui.widgets.banner import Banner  # noqa: E402
from tests.test_ovat_app_server import FakeAgent, FakeSession, _wf  # noqa: E402

SIZE = (120, 40)
# A path with no brand word in it, so a scan of the screen is about the app.
CWD = "/work/ovat-project"


class SavingAgent(FakeAgent):
    """An engine with conversation memory, for /save and /load."""

    def __init__(self, native=True):
        super().__init__(native)
        self.session = Session(system_prompt="be brief")


class AppFake(FakeSession):
    agent_class = SavingAgent


def run(coro):
    asyncio.run(asyncio.wait_for(coro, 90))


async def until(pilot, check, timeout=8.0):
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "condition not met in time"
        await pilot.pause(0.05)


async def type_text(pilot, text):
    for ch in text:
        await pilot.press("space" if ch == " " else ch)


async def say(pilot, text):
    """Type a line and send it."""
    await type_text(pilot, text)
    await pilot.press("enter")


def screen_text(app) -> str:
    """Everything the screen shows, as plain text (from the screenshot the
    app would export, so it is what a person would see)."""
    svg = app.export_screenshot()
    rows: dict[float, list[tuple[int, str]]] = {}
    for m in re.finditer(
            r'<text class="[^"]*" x="([\d.]+)" y="([\d.]+)" '
            r'textLength="([\d.]+)"[^>]*>(.*?)</text>', svg):
        x, y, width, raw = (float(m.group(1)), float(m.group(2)),
                            float(m.group(3)), unescape(m.group(4)))
        text = raw.replace("\xa0", " ")
        cell = width / max(1, len(text))
        rows.setdefault(y, []).append((round(x / cell), text))
    lines = []
    for y in sorted(rows):
        line = ""
        for col, text in sorted(rows[y]):
            line += " " * max(0, col - len(line)) + text
        lines.append(line.rstrip())
    return "\n".join(lines)


def assert_no_foreign_brand(app):
    text = screen_text(app).lower()
    for word in ("mistral", "vibe"):
        assert word not in text, f"{word!r} on screen:\n{screen_text(app)}"


def facts(app) -> str:
    return str(app.query_one("#agent-facts").render())


def banner_text(app) -> str:
    banner = app.query_one(Banner)
    return (str(banner._model.render()) + " | " + str(banner._counts.render()))


@pytest.fixture(autouse=True)
def copied(monkeypatch):
    """The clipboard is recorded, never touched: the real one writes to the
    terminal device."""
    seen = []
    monkeypatch.setattr(clipboard, "copy_to_clipboard",
                        lambda text: seen.append(text) or True)
    return seen


@pytest.fixture(autouse=True)
def refused(monkeypatch):
    """Every method the app asked and the server does not answer. A flow the
    person can reach must never be one of them, so each test ends by checking
    this list is empty (a test that WANTS one clears it first)."""
    from ovat.tui.server import rpc
    seen = []
    real = rpc.not_supported

    def spy(method):
        seen.append(method)
        return real(method)
    monkeypatch.setattr(rpc, "not_supported", spy)
    yield seen
    assert seen == [], f"the app asked for methods OVAT does not serve: {seen}"


@asynccontextmanager
async def running(size=SIZE, hardware=False, cwd=CWD, animate=False,
                  session_factory=None, ready="GPU", workflow="w.yml",
                  **script):
    """The app on the real server with a fake session. Always shut down;
    yields (app, pilot, session, handle). `session_factory` swaps the fake
    session for another (the real AgentSession over a fake agent)."""
    made = []

    def factory(workflow, sink, cwd=None):
        fake = (session_factory or AppFake)(workflow, sink, cwd, **script)
        made.append(fake)
        return fake

    session, handle = await start_ovat_app_server(
        workflow, session_factory=factory, cwd=cwd)
    app = OvatAgentApp(session, handle, history_file=None, animate=animate,
                       hardware=hardware)
    try:
        async with app.run_test(size=size, tooltips=True) as pilot:
            await until(pilot, lambda: app.query("#agent-facts")
                        and ready in facts(app))
            yield app, pilot, made[0], handle
    finally:
        await asyncio.wait_for(session.close(), 10)
        await asyncio.wait_for(handle.close(), 10)


# ---- the start -----------------------------------------------------------------

def test_the_app_starts_with_ovats_banner_and_bar():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await pilot.pause(0.3)
            banner = banner_text(app)
            assert "Qwen3.5-4B-int4-ov" in banner and "GPU" in banner
            assert "native" in banner and "1 tool" in banner
            assert "1 subagent" in banner and "loading" not in banner
            assert "GPU" in facts(app) and "native" in facts(app)
            text = screen_text(app)
            assert "OVAT" in text and str(Path(CWD)) in text
            assert app.theme == "ovat"          # "auto" is OVAT's own theme
            assert_no_foreign_brand(app)
    run(go())


def test_the_banner_shows_the_model_and_stops_saying_loading():
    # Seen with a real model: the banner's base on_mount ran twice (once
    # through super(), once through Textual's own dispatch), and the first
    # wave timer, no longer referenced, kept writing "loading" over the
    # model line after the agent was ready.
    async def go():
        async with running(animate=True, ready="",
                           hold_build=True) as (app, pilot, fake, handle):
            await pilot.pause(0.3)
            assert "loading" in banner_text(app)
            fake.build_gate.set()
            await until(pilot, lambda: handle.ready)
            await pilot.pause(0.5)
            assert "loading" not in banner_text(app)
            assert "Qwen3.5-4B-int4-ov" in banner_text(app)
    run(go())


def test_the_banner_shows_only_the_workflow_file_name():
    async def go():
        path = str(Path("/some/very/long/folder/subagents.yml"))
        async with running(workflow=path) as (app, pilot, fake, handle):
            banner = banner_text(app)
            assert "subagents.yml" in banner
            assert "long" not in banner
    run(go())


def test_nothing_that_phones_home_or_nags_is_on():
    """No update check, no telemetry, no feedback prompt, no editor promo,
    no what's-new from the app this was copied from, no notifications."""
    from ovat.tui.vibe.cli.textual_ui.widgets.feedback_bar import FeedbackBar
    from ovat.tui.vibe.cli.textual_ui.widgets.messages import (
        GreetingMessage, VscodeExtensionPromoMessage, WhatsNewMessage)

    async def go():
        async with running() as (app, pilot, fake, handle):
            await pilot.pause(0.5)
            config = app.config
            assert not config.enable_update_checks
            assert not config.enable_telemetry
            assert not config.enable_notifications
            assert not config.show_greeting
            assert app._update_notifier is None
            assert app._vscode_extension_promo is None
            assert not app._show_vscode_extension_promo
            assert not app.query(WhatsNewMessage)
            assert not app.query(VscodeExtensionPromoMessage)
            assert not app.query(GreetingMessage)
            assert not app.query_one(FeedbackBar).display
            assert (await app.app_server.resources.feedback.should_show(
                pending_user_messages=1)) is False
    run(go())


def test_without_a_workflow_the_app_says_how_to_start_one():
    async def go():
        async with running(workflow=None, session_factory=_real_session,
                           ready="") as (app, pilot, session, handle):
            await until(pilot, lambda: app.query(ErrorMessage))
            assert "No workflow here" in app.query_one(ErrorMessage)._error
            assert "/workflow" in app.query_one(ErrorMessage)._error
            assert app.is_running and not app._fatal_init_error
            await until(pilot, lambda: "no agent loaded" in banner_text(app))
    run(go())


def test_a_workflow_that_does_not_build_is_not_fatal():
    # The banner never gets an agent, so `running` cannot wait for its facts.
    async def failing():
        made = []

        def factory(workflow, sink, cwd=None):
            fake = AppFake(workflow, sink, cwd,
                           fail_build="Could not build w.yml: boom")
            made.append(fake)
            return fake
        session, handle = await start_ovat_app_server(
            "w.yml", session_factory=factory, cwd=CWD)
        app = OvatAgentApp(session, handle, history_file=None, animate=False,
                           hardware=False)
        try:
            async with app.run_test(size=SIZE, tooltips=True) as pilot:
                await until(pilot, lambda: app.query(ErrorMessage))
                assert "boom" in app.query_one(ErrorMessage)._error
                await until(pilot, lambda: "no agent loaded" in banner_text(
                    app))
                assert app.is_running and not app._fatal_init_error
                # ... and it is fixed the way the message says: /workflow.
                made[0].fail_build = None
                await say(pilot, "/workflow fixed.yml")
                await until(pilot, lambda: "GPU" in facts(app))
                assert "fixed.yml" in banner_text(app)
                await say(pilot, "hello")
                await until(pilot, lambda: app.query(AssistantMessage))
                assert made[0].asked == ["hello"]
        finally:
            await asyncio.wait_for(session.close(), 10)
            await asyncio.wait_for(handle.close(), 10)
    run(failing())


# ---- a turn --------------------------------------------------------------------

def test_a_question_streams_a_tool_group_then_the_answer():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "How many leave days?")
            await until(pilot, lambda: app.query(AssistantMessage))
            assert "How many leave days?" in app.query_one(
                UserMessage)._content
            assert app.query(ToolGroup)
            assert "New hires get 20 days." in app.query_one(
                AssistantMessage).get_content()
            await until(pilot, lambda: not app.query("LoadingWidget"))
            assert fake.asked == ["How many leave days?"]
            # the bar counts what the turn used: 100 + 20 tokens
            await until(pilot, lambda: "120 tokens" in facts(app))
            assert_no_foreign_brand(app)
    run(go())


def test_markup_like_text_cannot_break_answers_or_shell_output():
    async def go():
        async with running(answer="See C:\\hr\\[/leave].md [/bold]",
                           shell_lines=("hello [/bold] world",)) as (
                app, pilot, fake, handle):
            await say(pilot, "q")
            await until(pilot, lambda: app.query(AssistantMessage))
            assert "C:\\hr\\[/leave].md [/bold]" in app.query_one(
                AssistantMessage).get_content()
            await say(pilot, "!echo x")
            await until(pilot, lambda: "hello [/bold] world" in screen_text(
                app))
    run(go())


def test_esc_stops_a_running_answer():
    async def go():
        async with running(hold_turn=True) as (app, pilot, fake, handle):
            await say(pilot, "long")
            await until(pilot, lambda: fake.busy)
            await pilot.press("escape")
            await until(pilot, lambda: app.query(InterruptMessage))
            assert fake.cancels >= 1
            await until(pilot, lambda: not app._agent_job_active())
            # the prompt takes the next question
            fake.turn_gate.set()
            await say(pilot, "again")
            await until(pilot, lambda: fake.asked == ["long", "again"])
    run(go())


def test_a_question_typed_while_busy_waits_in_the_queue_and_runs_next():
    async def go():
        async with running(hold_turn=True) as (app, pilot, fake, handle):
            await say(pilot, "first")
            await until(pilot, lambda: fake.busy)
            await say(pilot, "second")
            await pilot.pause(0.3)
            assert fake.asked == ["first"]
            fake.turn_gate.set()
            await until(pilot, lambda: fake.asked == ["first", "second"])
    run(go())


# ---- approvals -----------------------------------------------------------------

@pytest.mark.parametrize("key, decision, said_yes", [
    ("1", ONCE, True), ("y", ONCE, True), ("2", SESSION, True),
    # 3 is "always allow" (saved for the project), 4 and n are Deny
    ("3", ALWAYS, True), ("4", REFUSE, False), ("n", REFUSE, False)])
def test_the_approval_panel_answers_reach_the_broker(
        monkeypatch, key, decision, said_yes):
    monkeypatch.setattr(approval_app, "_INPUT_GRACE_PERIOD_S", 0.0)

    async def go():
        async with running(approval={"task": "t"}) as (
                app, pilot, fake, handle):
            await say(pilot, "q")
            await until(pilot, lambda: app.query(ApprovalApp))
            panel = app.query_one(ApprovalApp)
            assert app.query_one(ChatInputContainer).display is False
            await until(pilot, lambda: app.focused is panel)
            assert_no_foreign_brand(app)
            await pilot.press(key)
            await until(pilot, lambda: fake.approved is not None)
            assert fake.approved is said_yes
            assert [d for _, d in fake.resolved] == [decision]
            await until(pilot, lambda: not app.query(ApprovalApp))
            assert app.query_one(ChatInputContainer).display is True
    run(go())


def test_the_approval_panel_ignores_keys_inside_its_grace_period(monkeypatch):
    monkeypatch.setattr(approval_app, "_INPUT_GRACE_PERIOD_S", 60.0)

    async def go():
        async with running(approval={"task": "t"}) as (
                app, pilot, fake, handle):
            await say(pilot, "q")
            await until(pilot, lambda: app.query(ApprovalApp))
            await until(pilot, lambda: app.focused is app.query_one(
                ApprovalApp))
            await pilot.press("1")
            await pilot.pause(0.3)
            assert fake.approved is None and fake.resolved == []
            fake.broker.cancel_all()
    run(go())


def test_approval_text_cannot_dress_up_as_the_panels_own_options(monkeypatch):
    monkeypatch.setattr(approval_app, "_INPUT_GRACE_PERIOD_S", 60.0)
    spoof = "summarise\n\n› 1. Allow once\n  3. Deny\x1b[2J"

    async def go():
        async with running(approval={"task": spoof}) as (
                app, pilot, fake, handle):
            await say(pilot, "q")
            await until(pilot, lambda: app.query(ApprovalApp))
            panel = app.query_one(ApprovalApp)
            await until(pilot, lambda: "summarise" in "".join(
                str(w.render()) for w in panel.query("NoMarkupStatic")))
            shown = " ".join(str(w.render()) for w in panel.query(
                ".approval-description"))
            assert shown, "the task must be on screen to be judged"
            assert "\u21b5" in shown and "\n" not in shown
            assert "\x1b" not in shown and "\\x1b" not in shown
            fake.broker.cancel_all()
    run(go())


def test_esc_closes_an_open_approval_with_a_no(monkeypatch):
    monkeypatch.setattr(approval_app, "_INPUT_GRACE_PERIOD_S", 0.0)

    async def go():
        async with running(approval={"task": "t"}) as (
                app, pilot, fake, handle):
            await say(pilot, "q")
            await until(pilot, lambda: app.query(ApprovalApp))
            await until(pilot, lambda: app.focused is app.query_one(
                ApprovalApp))
            await pilot.press("escape")
            await until(pilot, lambda: fake.approved is not None)
            assert fake.approved is False
    run(go())


# ---- ! --------------------------------------------------------------------------

def test_bang_runs_a_shell_command_and_shows_its_output_once():
    async def go():
        async with running(shell_lines=("one", "two")) as (
                app, pilot, fake, handle):
            await say(pilot, "!echo hi")
            await until(pilot, lambda: "two" in screen_text(app))
            text = screen_text(app)
            assert "Ran echo hi" in text and text.count("one") == 1
            assert not app.query(ErrorMessage)
            assert fake.shell_busy is False
    run(go())


def test_esc_stops_a_running_shell_command():
    async def go():
        async with running(hold_shell=True) as (app, pilot, fake, handle):
            await say(pilot, "!sleep 100")
            await until(pilot, lambda: fake.shell_busy)
            await pilot.press("escape")
            await until(pilot, lambda: fake.shell_killed.is_set())
            await until(pilot, lambda: not app._is_busy())
    run(go())


def test_an_empty_bang_says_what_to_type():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "!")
            await until(pilot, lambda: app.query(ErrorMessage))
    run(go())


# ---- the commands --------------------------------------------------------------

def test_help_lists_the_commands_and_keys_and_nothing_that_cannot_work():
    async def go():
        async with running(size=(120, 60)) as (app, pilot, fake, handle):
            await say(pilot, "/help")
            await until(pilot, lambda: len(app.query(UserCommandMessage)))
            await pilot.pause(0.3)
            text = screen_text(app)
            for name in [c.name for c in ovat_commands.COMMANDS
                         if c.name not in ("quit", "new")] + [
                             "clear", "exit", "status", "theme", "resume",
                             "rewind", "rename", "branch", "log", "compact",
                             "model", "thinking", "reload", "config", "mcp"]:
                assert f"/{name}" in text, name
            for gone in ("/teleport", "/retry"):
                assert gone not in text, gone
            assert "Ctrl+O" in text and "Shift+Tab" in text
            # /branch's own text has `ovat agent --resume <id>` in a code
            # span, where an escape would be shown as is
            assert "&lt;" not in text and "--resume <id>" in text
            # a usage in angle brackets is not eaten as an HTML tag
            assert "<engine>" in text and "<subagent> <task>" in text
            assert_no_foreign_brand(app)
    run(go())


def test_slash_lists_every_command_in_the_popup():
    async def go():
        async with running(size=(120, 50)) as (app, pilot, fake, handle):
            await pilot.press("/")
            popup = app.query_one("#completion-popup")
            await until(pilot, lambda: popup.styles.display == "block")
            text = popup.content_text
            for name in ("help", "doctor", "telemetry", "agents", "ask"):
                assert f"/{name}" in text
            await pilot.press("escape")
    run(go())


def _highlighted(popup) -> int:
    rows = list(popup.query("_CompletionRow"))
    return next(i for i, r in enumerate(rows)
                if r.has_class("completion-selected"))


def test_the_wheel_moves_the_highlight_in_the_completion_popup():
    from textual import events

    async def go():
        async with running(size=(120, 50)) as (app, pilot, fake, handle):
            await pilot.press("/")
            popup = app.query_one("#completion-popup")
            await until(pilot, lambda: popup.styles.display == "block")
            assert _highlighted(popup) == 0
            popup.post_message(events.MouseScrollDown(
                popup, 1, 1, 0, 1, 0, False, False, False))
            await until(pilot, lambda: _highlighted(popup) == 1)
            popup.post_message(events.MouseScrollUp(
                popup, 1, 1, 0, -1, 0, False, False, False))
            await until(pilot, lambda: _highlighted(popup) == 0)
            await pilot.press("escape")
    run(go())


def test_a_click_on_a_row_of_the_completion_popup_chooses_it():
    async def go():
        async with running(size=(120, 50)) as (app, pilot, fake, handle):
            await pilot.press("/")
            popup = app.query_one("#completion-popup")
            await until(pilot, lambda: popup.styles.display == "block")
            rows = list(popup.query("_CompletionRow"))
            label = str(rows[2].query_one(".completion-command").render())
            await pilot.click(rows[2].query_one(".completion-command"))
            box = app.query_one("#input-body")
            await until(pilot, lambda: box.value.strip() == label.strip())
            assert popup.styles.display == "none"
    run(go())


def test_at_offers_the_subagents_before_any_file():
    async def go():
        async with running(size=(120, 50)) as (app, pilot, fake, handle):
            await type_text(pilot, "@resear")
            popup = app.query_one("#completion-popup")
            await until(pilot, lambda: popup.styles.display == "block")
            first = popup.query("_CompletionRow").first()
            assert "researcher" in str(
                first.query_one(".completion-command").render())
            assert "reads docs" in popup.content_text
            await pilot.press("tab")
            box = app.query_one("#input-body")
            await until(pilot, lambda: box.value.startswith("@researcher"))
    run(go())


def test_an_unknown_command_suggests_a_near_one_and_is_not_sent_to_the_model():
    async def go():
        async with running() as (app, pilot, fake, handle):
            # (typed, the completion list would take the first Enter)
            await app._dispatch_submitted_value("/tols x")
            await until(pilot, lambda: app.query(WarningMessage))
            text = app.query_one(WarningMessage)._message
            assert "/tols" in text and "/tools" in text
            await say(pilot, "/zzqq")
            await until(pilot, lambda: len(app.query(WarningMessage)) == 2)
            assert "No command `/zzqq`" in list(
                app.query(WarningMessage))[1]._message
            assert fake.asked == []
            # a path is not a mistyped command: it is a question
            await say(pilot, "/etc/hosts what is this")
            await until(pilot, lambda: fake.asked)
            assert fake.asked == ["/etc/hosts what is this"]
    run(go())


def test_new_clears_the_conversation_and_clear_does_too():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "one")
            await until(pilot, lambda: app.query(AssistantMessage))
            await say(pilot, "/new")
            await until(pilot, lambda: fake.cleared == 1)
            await until(pilot, lambda: not app.query(AssistantMessage))
            assert "New conversation started" in screen_text(app)
            await say(pilot, "/clear")
            await until(pilot, lambda: fake.cleared == 2)
    run(go())


def test_tools_lists_the_tools_with_their_permissions():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "/tools")
            await until(pilot, lambda: "search_docs" in screen_text(app))
            text = screen_text(app)
            assert "delegate" in text and "allow" in text
    run(go())


def test_agents_lists_subagents_and_whether_a2a_is_serving():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "/agents")
            await until(pilot, lambda: "researcher" in screen_text(app))
            text = screen_text(app)
            assert "reads docs" in text and "not serving" in text
    run(go())


def test_ask_sends_the_task_straight_to_the_subagent():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "/ask researcher find the leave rules")
            await until(pilot, lambda: fake.subagent_calls)
            assert fake.subagent_calls == [
                ("researcher", "find the leave rules")]
            assert fake.asked == []                  # not the main model
            await until(pilot, lambda: app.query(AssistantMessage))
            # one line for what was asked, not the command AND an @ line
            assert not app.query(SlashCommandMessage)
            assert [m._content for m in app.query(UserMessage)] == [
                "@researcher find the leave rules"]
    run(go())


def test_ask_needs_a_known_subagent_and_a_task():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "/ask researcher")
            await until(pilot, lambda: len(app.query(WarningMessage)) == 1)
            assert "Usage" in app.query_one(WarningMessage)._message
            await say(pilot, "/ask nobody do it")
            await until(pilot, lambda: len(app.query(WarningMessage)) == 2)
            assert "researcher" in list(app.query(WarningMessage))[1]._message
            assert fake.subagent_calls == [] and fake.asked == []
    run(go())


def test_serve_starts_and_stops_the_a2a_server_and_its_lines_are_notes():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "/serve")
            await until(pilot, lambda: fake.server_calls == ["start"])
            # the server's line is a note in the transcript, not a toast
            await until(pilot, lambda: any(
                "serving on 127.0.0.1:9999" in screen_text(app)
                for _ in [0]))
            assert not any("serving on" in n.message
                           for n in app._notifications)
            await until(pilot, lambda: "a2a serving" in facts(app))
            await say(pilot, "/serve stop")
            await until(pilot, lambda: fake.server_calls == ["start", "stop"])
            await until(pilot, lambda: "a2a serving" not in facts(app))
    run(go())


def test_notices_from_the_session_land_in_the_transcript_by_level():
    async def go():
        async with running() as (app, pilot, fake, handle):
            fake.sink(s.Notice("warn", "a2a server stopped (exit 1)"))
            fake.sink(s.Notice("error", "it broke"))
            await until(pilot, lambda: app.query(WarningMessage)
                        and app.query(ErrorMessage))
            assert "a2a server stopped" in app.query_one(
                WarningMessage)._message
            assert not app._notifications
    run(go())


def test_engine_switches_rebuild_the_agent_and_the_screen_follows():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "/engine fast")
            await until(pilot, lambda: app.query(WarningMessage))
            assert "native, react" in app.query_one(WarningMessage)._message
            assert fake.engines == [None]
            await say(pilot, "/engine react")
            await until(pilot, lambda: fake.engines == [None, "react"])
            await until(pilot, lambda: "react" in facts(app))
            await until(pilot, lambda: "react" in banner_text(app))
            assert "loading" not in banner_text(app)
            # and the next question runs on the new agent
            await say(pilot, "hi")
            await until(pilot, lambda: fake.asked == ["hi"])
    run(go())


def test_workflow_loads_another_file_and_a_bad_one_keeps_the_old_agent():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "/workflow other.yml")
            await until(pilot, lambda: fake.workflow == "other.yml"
                        and "other.yml" in banner_text(app))
            await say(pilot, "/workflow bad.yml")
            await until(pilot, lambda: app.query(ErrorMessage))
            assert "bad.yml" in app.query_one(ErrorMessage)._error
            await until(pilot, lambda: fake.workflow == "other.yml")
            await until(pilot, lambda: "GPU" in banner_text(app)
                        and "other.yml" in banner_text(app))
            await say(pilot, "still there")
            await until(pilot, lambda: fake.asked == ["still there"])
            await say(pilot, "/workflow")
            await until(pilot, lambda: len(app.query(WarningMessage)) >= 1)
    run(go())


def test_switching_is_refused_politely_while_the_agent_works():
    async def go():
        async with running(hold_turn=True) as (app, pilot, fake, handle):
            await say(pilot, "long")
            await until(pilot, lambda: fake.busy)
            await say(pilot, "/engine react")
            await pilot.pause(0.3)
            assert fake.engines == [None]        # not rebuilt under a turn
            fake.turn_gate.set()
    run(go())


def test_trace_says_what_the_last_answer_did():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "/trace")
            await until(pilot, lambda: "Nothing yet" in screen_text(app))
            fake.last_trace = {
                "engine": "native",
                "totals": {"turns": 2, "tool_calls": 1, "latency_s": 1.5},
                "turns": [{"latency_s": 1.0, "prompt_tokens": 90,
                           "completion_tokens": 10, "finish_reason": "stop",
                           "tool_calls": [{"name": "search_docs",
                                           "duration_s": 0.4}]}]}
            await say(pilot, "/trace")
            await until(pilot, lambda: "search_docs" in screen_text(app))
            assert "2 turns" in screen_text(app)
    run(go())


def test_yolo_switches_to_auto_approve_and_back(monkeypatch):
    monkeypatch.setattr(approval_app, "_INPUT_GRACE_PERIOD_S", 0.0)

    async def go():
        async with running(approval={"task": "t"}) as (
                app, pilot, fake, handle):
            await say(pilot, "/yolo")
            await until(pilot, lambda: fake.broker.auto_approve)
            await until(pilot, lambda: "auto-approve" in screen_text(app))
            # a tool that would ask is now approved without the panel
            await say(pilot, "q")
            await until(pilot, lambda: fake.approved is not None)
            assert fake.approved is True and not app.query(ApprovalApp)
            await until(pilot, lambda: not app._is_busy())
            await say(pilot, "/yolo")
            await until(pilot, lambda: not fake.broker.auto_approve)
            await until(pilot, lambda: "Tools ask again" in screen_text(app))
    run(go())


def test_shift_tab_does_the_same_switch_as_yolo():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await pilot.press("shift+tab")
            await until(pilot, lambda: fake.broker.auto_approve)
            await pilot.press("shift+tab")
            await until(pilot, lambda: not fake.broker.auto_approve)
    run(go())


def test_save_and_load_keep_a_conversation(tmp_path):
    async def go():
        async with running(cwd=str(tmp_path)) as (app, pilot, fake, handle):
            await say(pilot, "/save mine")
            saved = tmp_path / ".ovat" / "sessions" / "mine.json"
            await until(pilot, lambda: saved.is_file())
            fake.agent.session = Session(system_prompt="other")
            await say(pilot, "/load mine")
            await until(pilot, lambda: fake.agent.session.messages
                        == Session.load(str(saved)).messages
                        and "Loaded" in screen_text(app))
            await say(pilot, "/load nothere")
            await until(pilot, lambda: "Saved: mine" in " ".join(
                w._message for w in app.query(WarningMessage)))
    run(go())


def test_copy_copies_the_last_answer(copied):
    async def go():
        async with running(answer="Twenty days.") as (
                app, pilot, fake, handle):
            await say(pilot, "/copy")
            await pilot.pause(0.3)
            assert copied == []                  # nothing to copy yet
            await say(pilot, "q")
            await until(pilot, lambda: app.query(AssistantMessage))
            await say(pilot, "/copy")
            await until(pilot, lambda: copied)
            assert "Twenty days." in copied[-1]
    run(go())


def test_status_shows_the_agents_own_numbers():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "q")
            await until(pilot, lambda: app.query(AssistantMessage))
            await say(pilot, "/status")
            await until(pilot, lambda: "Agent Statistics" in screen_text(app))
            text = screen_text(app)
            assert "Qwen3.5-4B-int4-ov on GPU, native engine" in text
            assert "Session Prompt Tokens" in text and "100" in text
            assert "$" not in text                # a local model has no bill
            assert_no_foreign_brand(app)
    run(go())


def test_quit_leaves_the_app():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "/quit")
            await until(pilot, lambda: not app.is_running)
    run(go())


def test_theme_saves_the_choice_and_esc_restores_the_previous_theme(tmp_path):
    from ovat.agentic import prefs

    async def go():
        async with running(cwd=str(tmp_path)) as (app, pilot, fake, handle):
            assert app.theme == "ovat"
            await say(pilot, "/theme")
            await until(pilot, lambda: app.query(ThemePickerApp))
            picker = app.query_one(ThemePickerApp)
            names = [str(o.prompt) for o in picker.query_one(
                "OptionList").options] if picker.query("OptionList") else []
            assert any("ovat" in n for n in names)
            await pilot.press("down")
            await until(pilot, lambda: app.theme != "ovat")
            await pilot.press("escape")
            await until(pilot, lambda: not app.query(ThemePickerApp))
            assert app.theme == "ovat"
            await say(pilot, "/theme")
            await until(pilot, lambda: app.query(ThemePickerApp))
            await pilot.press("down")
            await pilot.press("enter")
            await until(pilot, lambda: not app.query(ThemePickerApp))
            chosen = app.theme
            assert chosen != "ovat"
            await until(pilot, lambda: prefs.load(str(tmp_path)).get("theme") == chosen)
            # A later runtime update keeps the saved choice.
            await say(pilot, "/engine react")
            await until(pilot, lambda: "react" in facts(app))
            await pilot.pause(0.3)
            assert app.theme == chosen
    run(go())


# ---- /doctor and /telemetry -----------------------------------------------------

def _plain(widgets) -> str:
    import io

    from rich.console import Console
    out = io.StringIO()
    console = Console(file=out, width=100, color_system=None)
    for widget in widgets:
        if all(node.display for node in widget.ancestors_with_self
               if hasattr(node, "display")):
            console.print(widget.content)
    return "".join(out.getvalue().split())


def _fake_checks(config_path=None):
    from ovat.cli.diagnostics import Check
    return [Check("Python", "ok", "fine here"),
            Check("OVMS serving", "warn", "Run 'ovat setup' to install it"),
            Check("Core dependencies", "ok", "all 9 importable"),
            Check("Workflow config", "fail", "no such file: w.yml")]


def test_doctor_is_a_block_in_the_chat_and_ctrl_o_opens_what_passed(
        monkeypatch):
    from ovat.cli import diagnostics
    from ovat.tui.widgets.doctor import DoctorReport
    monkeypatch.setattr(diagnostics, "run_checks", _fake_checks)

    async def go():
        async with running(size=(100, 40)) as (app, pilot, fake, handle):
            await say(pilot, "/doctor")
            await until(pilot, lambda: app.query(DoctorReport)
                        and app.query_one(DoctorReport).checks is not None)
            await pilot.pause(0.2)
            report = app.query_one(DoctorReport)
            assert report.screen is app.screen              # in the chat
            shown = _plain(report.query("Static"))
            assert shown.index("Workflowconfig") < shown.index("OVMSserving")
            assert "2checksok" in shown and "finehere" not in shown
            await pilot.press("ctrl+o")
            await pilot.pause(0.2)
            assert "all9importable" in _plain(report.query("Static"))
            assert_no_foreign_brand(app)
    run(go())


def test_doctor_copy_puts_the_plain_report_on_the_clipboard(
        monkeypatch, copied):
    from ovat.cli import diagnostics
    from ovat.tui.widgets.doctor import DoctorReport
    monkeypatch.setattr(diagnostics, "run_checks", _fake_checks)

    async def go():
        async with running(size=(100, 30)) as (app, pilot, fake, handle):
            await say(pilot, "/doctor copy")
            await until(pilot, lambda: "run `/doctor` first" in screen_text(
                app).replace("\n", " ") or "Nothing to copy" in screen_text(
                app))
            await say(pilot, "/doctor")
            await until(pilot, lambda: app.query(DoctorReport)
                        and app.query_one(DoctorReport).checks is not None)
            await say(pilot, "/doctor copy")
            await until(pilot, lambda: copied)
            lines = copied[-1].splitlines()
            assert "1 fail · 1 warn · 2 ok" in lines[0]
    run(go())


def test_a_doctor_that_cannot_run_says_why_and_the_app_lives(monkeypatch):
    from ovat.cli import diagnostics
    from ovat.tui.widgets.doctor import DoctorReport

    def broken(config_path=None):
        raise ImportError("openvino is broken")
    monkeypatch.setattr(diagnostics, "run_checks", broken)

    async def go():
        async with running(size=(100, 30)) as (app, pilot, fake, handle):
            await say(pilot, "/doctor")
            await until(pilot, lambda: app.query(DoctorReport)
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


def test_telemetry_opens_over_the_chat_with_the_agents_numbers_and_esc_returns(
        monkeypatch):
    from ovat.tui import telemetry
    sources = _telemetry_sources()
    monkeypatch.setattr(telemetry, "_default_sources", lambda: sources)

    async def go():
        async with running(size=(80, 24), hold_turn=True) as (
                app, pilot, fake, handle):
            fake.agent.last_trace = {"totals": {
                "prompt_tokens": 120, "completion_tokens": 30, "turns": 2}}
            await say(pilot, "long")             # a turn is running: Esc on
            await until(pilot, lambda: fake.busy)  # the page must not stop it
            await say(pilot, "/telemetry")
            await until(pilot, lambda: isinstance(app.screen,
                                                   telemetry.TelemetryView)
                        and app.screen._tiles)
            view = app.screen
            assert view.styles.background.a == 1            # opaque
            await until(pilot, lambda: view.live.latest(
                "agent.prompt_tokens") == 120)
            assert "npu n/a (no NPU found)" in str(
                view.query_one("#tel-sources").render())
            await pilot.press("escape")
            await until(pilot, lambda: not isinstance(
                app.screen, telemetry.TelemetryView))
            assert all(src.stopped for src in sources)
            assert fake.cancels == 0              # Esc closed the page only
            assert fake.busy
            fake.turn_gate.set()
    run(go())


# ---- keys ----------------------------------------------------------------------

def test_ctrl_o_folds_and_opens_the_tool_calls():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "q")
            await until(pilot, lambda: app.query(AssistantMessage))
            group = app.query_one(ToolGroup)
            assert group._is_collapsed
            await pilot.press("ctrl+o")
            await until(pilot, lambda: not group._is_collapsed)
            await pilot.press("ctrl+o")
            await until(pilot, lambda: group._is_collapsed)
    run(go())


def test_ctrl_t_also_folds_the_tool_calls_for_terminals_that_eat_ctrl_o():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "q")
            await until(pilot, lambda: app.query(AssistantMessage))
            group = app.query_one(ToolGroup)
            await pilot.press("ctrl+t")
            await until(pilot, lambda: not group._is_collapsed)
            await pilot.press("ctrl+t")
            await until(pilot, lambda: group._is_collapsed)
    run(go())


def test_ctrl_c_twice_quits_when_idle():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await pilot.press("ctrl+c")
            assert app.is_running
            assert "again to quit" in screen_text(app)
            await pilot.press("ctrl+c")
            await until(pilot, lambda: not app.is_running)
    run(go())


def test_ctrl_c_stops_a_running_answer_and_does_not_quit():
    async def go():
        async with running(hold_turn=True) as (app, pilot, fake, handle):
            await say(pilot, "long")
            await until(pilot, lambda: fake.busy)
            await pilot.press("ctrl+c")
            await until(pilot, lambda: fake.cancels >= 1)
            await until(pilot, lambda: app.query(InterruptMessage))
            assert app.is_running
    run(go())


def test_ctrl_c_copies_a_selection_and_then_counts_towards_quit(copied):
    async def go():
        async with running(answer="Twenty days of leave.") as (
                app, pilot, fake, handle):
            await say(pilot, "q")
            await until(pilot, lambda: app.query(AssistantMessage))
            await pilot.pause(0.3)
            app.screen.text_select_all()          # as a drag with the mouse
            assert "Twenty days of leave." in app.screen.get_selected_text()
            await pilot.press("ctrl+c")
            await until(pilot, lambda: copied)
            assert "Twenty days of leave." in copied[-1]
            assert not app.screen.get_selected_text()    # cleared by the copy
            assert app.is_running and fake.cancels == 0
            await pilot.press("ctrl+c")           # nothing selected now: the
            await pilot.pause(0.2)                # first press of two
            assert app.is_running
            await pilot.press("ctrl+c")
            await until(pilot, lambda: not app.is_running)
    run(go())


def test_ctrl_enter_steers_a_waiting_prompt_into_the_running_turn():
    """A prompt typed while the agent works waits in the queue. Ctrl+Enter
    (or Enter on an empty prompt) hands it to the running turn instead of
    waiting for the next one."""
    async def go():
        async with running(hold_turn=True) as (app, pilot, fake, handle):
            await say(pilot, "first")
            await until(pilot, lambda: fake.busy)
            await say(pilot, "second")                  # waits in the queue
            await until(pilot, lambda: "second" in screen_text(app))
            await pilot.press("ctrl+enter")
            await until(pilot, lambda: fake.steered == ["second"])
            assert not app.query(ErrorMessage)
            fake.turn_gate.set()
            await until(pilot, lambda: not fake.busy)
            # it joined the first turn; it was not asked as a turn of its own
            assert fake.asked == ["first"]
            assert "second" in screen_text(app)
    run(go())


def test_ctrl_enter_on_a_framework_engine_says_why_and_the_prompt_still_runs():
    async def go():
        async with running(hold_turn=True, native=False) as (
                app, pilot, fake, handle):
            await say(pilot, "first")
            await until(pilot, lambda: fake.busy)
            await say(pilot, "second")
            await until(pilot, lambda: "second" in screen_text(app))
            await pilot.press("ctrl+enter")
            await until(pilot, lambda: "native engine" in screen_text(app))
            fake.turn_gate.set()
            await until(pilot, lambda: fake.asked == ["first", "second"])
    run(go())


def test_the_debug_console_key_does_nothing_instead_of_failing():
    """Ctrl+Backslash (the debug console) is a key of the copied app whose
    server side OVAT does not have. It must not reach the server."""
    from ovat.tui.vibe.cli.textual_ui.widgets.debug_console import DebugConsole

    async def go():
        async with running() as (app, pilot, fake, handle):
            await pilot.press("ctrl+backslash")
            await pilot.pause(0.3)
            assert not app.query(DebugConsole) and not app.query(ErrorMessage)
    run(go())


def test_esc_esc_rewinds_to_a_message_and_puts_it_back_in_the_prompt(
        monkeypatch):
    from ovat.tui.vibe.cli.textual_ui import app as vibe_app
    from ovat.tui.vibe.cli.textual_ui.widgets.rewind_app import RewindApp
    # "Esc Esc" is two presses within a fraction of a second; a slow runner
    # cannot be trusted to be that quick, so the window is made wide.
    monkeypatch.setattr(vibe_app, "DOUBLE_ESC_DELAY", 30.0)

    async def go():
        async with running() as (app, pilot, fake, handle):
            for i, text in enumerate(("first", "second")):
                await say(pilot, text)
                await until(pilot, lambda: len(app.query(AssistantMessage))
                            == i + 1)
                await until(pilot, lambda: not app._is_busy())
            await pilot.press("escape")
            await pilot.press("escape")
            await until(pilot, lambda: app.query(RewindApp))
            # the last message is picked; "edit only" (OVAT restores no
            # files), then "keep in current session"
            await pilot.press("1")
            await pilot.press("1")
            await until(pilot, lambda: not app._rewind_mode
                        and len(app.query(UserMessage)) == 1)
            assert app.query_one(ChatInputContainer).value == "second"
            assert [m["content"] for m in fake.remembered.messages
                    if m["role"] == "user"] == ["first"]
            assert not app.query(ErrorMessage)
    run(go())


def test_compact_summarizes_the_memory_and_marks_the_transcript():
    from ovat.tui.vibe.cli.textual_ui.widgets.compact import CompactMessage

    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "first")
            await until(pilot, lambda: len(app.query(AssistantMessage)) == 1
                        and not app._is_busy())
            await say(pilot, "/compact keep the numbers")
            await until(pilot, lambda: app.query(CompactMessage)
                        and "Compaction completed."
                        in app.query_one(CompactMessage).get_content())
            assert fake.compacts == ["keep the numbers"]
            assert [e.kind for e in app.app_server.history
                    if e.type == "checkpoint"] == ["compaction"]
            assert len(fake.memory()) == 3      # system, summary, ack
            assert not app.query(ErrorMessage), [
                e._error for e in app.query(ErrorMessage)]
            # and the next question goes on from the summary
            await say(pilot, "second")
            await until(pilot, lambda: fake.asked == ["first", "second"]
                        and not app._is_busy())
            assert not app.query(ErrorMessage), [
                e._error for e in app.query(ErrorMessage)]
    run(go())


def test_model_picker_switches_the_model_and_the_conversation_is_kept(tmp_path):
    from ovat.tui.vibe.cli.textual_ui.widgets.model_picker import \
        ModelPickerApp

    async def go():
        async with running(workflow=_wf(tmp_path)) as (app, pilot, fake, handle):
            await say(pilot, "first")
            await until(pilot, lambda: fake.asked == ["first"]
                        and not app._is_busy())
            await say(pilot, "/model")
            await until(pilot, lambda: app.query(ModelPickerApp))
            # rows: Default, then the models in the order the server lists
            await pilot.press("down", "down", "enter")
            await until(pilot, lambda: fake.picks == [(None, None)])
            await until(pilot, lambda: "Qwen3-8B-int4-ov" in facts(app))
            await until(pilot, lambda: "Qwen3-8B-int4-ov" in banner_text(app))
            assert not app.query(ErrorMessage), [
                e._error for e in app.query(ErrorMessage)]
            # the model remembers the first question, and a second one runs
            await say(pilot, "second")
            await until(pilot, lambda: fake.asked == ["first", "second"]
                        and not app._is_busy())
            assert [m["content"] for m in fake.memory()
                    if m["role"] == "user"] == ["first", "second"]
    run(go())


def test_model_picker_default_row_goes_back_to_the_workflows_model(tmp_path):
    from ovat.tui.vibe.cli.textual_ui.widgets.model_picker import \
        ModelPickerApp

    async def go():
        async with running(workflow=_wf(tmp_path)) as (app, pilot, fake, handle):
            await fake_pick(app, pilot, ["down", "down", "enter"])
            await until(pilot, lambda: "Qwen3-8B-int4-ov" in facts(app))
            await say(pilot, "/model")
            await until(pilot, lambda: app.query(ModelPickerApp))
            await pilot.press("home", "enter")           # the Default row
            await until(pilot, lambda: fake.picks[-1] == (None, None))
            await until(pilot, lambda: "Qwen3-8B-int4-ov" in facts(app))
    run(go())


async def fake_pick(app, pilot, keys):
    await say(pilot, "/model")
    await until(pilot, lambda: app.query("#modelpicker-app"))
    await pilot.press(*keys)


def test_thinking_picker_switches_thinking_and_the_bar_says_so(tmp_path):
    from ovat.tui.vibe.cli.textual_ui.widgets.thinking_picker import \
        ThinkingPickerApp

    async def go():
        async with running(workflow=_wf(tmp_path)) as (app, pilot, fake, handle):
            await say(pilot, "first")
            await until(pilot, lambda: fake.asked == ["first"]
                        and not app._is_busy())
            await say(pilot, "/thinking")
            await until(pilot, lambda: app.query(ThinkingPickerApp))
            await pilot.press("down", "enter")          # off -> high
            await until(pilot, lambda: fake.picks == [(None, None)])
            await until(pilot, lambda: "thinking" in facts(app))
            assert [m["content"] for m in fake.memory()
                    if m["role"] == "user"] == ["first"]
            assert not app.query(ErrorMessage)
    run(go())


def test_reload_says_it_is_done_and_builds_nothing_when_the_file_is_the_same():
    from ovat.tui.vibe.cli.textual_ui.widgets.reload_message import \
        ReloadConfigMessage

    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "/reload")
            await until(pilot, lambda: app.query(ReloadConfigMessage))
            await pilot.pause(0.3)
            assert not app.query(ErrorMessage)
            assert fake.picks == []
    run(go())


def test_saved_conversations_rename_branch_log_and_resume_from_the_app(
        saved_sessions):
    from ovat.tui.vibe.cli.textual_ui.widgets.branch_created_message import \
        BranchCreatedMessage
    from ovat.tui.vibe.cli.textual_ui.widgets.session_picker import \
        SessionPickerApp

    def user_texts(app):
        return [w._content for w in app.query(UserMessage)]

    # (slow on a busy Windows runner: one 8 s wait timed out under -n 4)
    async def go():
        async with running(size=(120, 40)) as (app, pilot, fake, handle):
            await say(pilot, "first chat")
            await until(pilot, timeout=30, check=lambda: app.query(AssistantMessage))
            await until(pilot, timeout=30, check=lambda: not app._is_busy())
            first = app.app_server.session_id
            await say(pilot, "/rename Leave policy")
            await until(pilot, timeout=30, check=lambda: (saved_sessions.load(first) or {})
                        .get("title") == "Leave policy")
            await say(pilot, "/log")
            # The folder is checked in the message, not on screen: on Windows
            # CI it never showed up in the screen text, however long the test
            # waited (cause not verified; a long temp path wrapping mid-name
            # is the likely one, it did not reproduce on macOS).
            await until(pilot, timeout=30, check=lambda: any(
                str(saved_sessions.folder) in m._content
                for m in app.query(UserCommandMessage)))
            assert "Current Log Directory" in screen_text(app)
            await say(pilot, "/branch")
            await until(pilot, timeout=30, check=lambda: app.query(BranchCreatedMessage))
            assert len(saved_sessions.list()) == 2
            await say(pilot, "/new")
            await until(pilot, timeout=30, check=lambda: app.app_server.session_id != first)
            await say(pilot, "second chat")
            # (/new leaves its own "clear" line above it)
            await until(pilot, timeout=30, check=lambda: user_texts(app)[-1:] == ["second chat"]
                        and not app._is_busy())
            await say(pilot, "/resume")
            await until(pilot, timeout=30, check=lambda: app.query(SessionPickerApp))
            picker = app.query_one(SessionPickerApp)
            options = picker.query_one("OptionList")
            # this chat, the branch and "Leave policy". The branch and its
            # source are saved in the same instant, so pick by id, not place
            await until(pilot, timeout=30, check=lambda: options.option_count == 3)
            options.highlighted = options.get_option_index(first)
            await pilot.press("enter")
            await until(pilot, timeout=30, check=lambda: app.app_server.session_id == first)
            await until(pilot, timeout=30, check=lambda: user_texts(app) == ["first chat"])
            assert [m["content"] for m in fake.remembered.messages
                    if m["role"] == "user"] == ["first chat"]
            assert not app.query(ErrorMessage)
    run(go())


# ---- the screen ------------------------------------------------------------------

def test_the_prompt_stays_on_screen_at_80x24_even_with_a_long_conversation():
    async def go():
        async with running(size=(80, 24)) as (app, pilot, fake, handle):
            area = app.query_one(ChatInputContainer).input_widget
            assert 0 <= area.region.y < 24 and area.region.height >= 1
            for i in range(4):
                await say(pilot, f"question {i}")
                await until(pilot, lambda: len(app.query(AssistantMessage))
                            == i + 1)
            area = app.query_one(ChatInputContainer).input_widget
            assert 0 <= area.region.y < 24 and area.region.height >= 1
            assert app.screen.size.height == 24
            assert_no_foreign_brand(app)
    run(go())


def test_the_bottom_bar_fits_at_80_columns():
    async def go():
        async with running(size=(80, 24)) as (app, pilot, fake, handle):
            bar = app.query_one("#bottom-bar")
            assert bar.region.right <= 80
            assert app.query_one("#agent-facts").region.right <= 80
    run(go())


def test_there_are_no_tooltips():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "/help")
            await pilot.pause(0.3)
            assert all(w.tooltip is None for w in app.screen.walk_children())
    run(go())


def test_the_bar_shows_a_live_hardware_reading_without_inventing_one():
    class Monitor:
        closed = False

        def sample(self):
            return {"cpu": 23.4, "ram": 61.0}

        def close(self):
            self.closed = True
    monitor = Monitor()

    async def go():
        async with running(size=(160, 30), hardware=monitor) as (
                app, pilot, fake, handle):
            await until(pilot, lambda: "CPU 23%" in facts(app), timeout=8)
            assert "RAM 61%" in facts(app) and "NPU" not in facts(app)
    run(go())
    assert monitor.closed


def test_the_banner_mark_animates_then_rests():
    from ovat.tui.widgets.braille import ChipMark, frames

    async def go():
        async with running(animate=True) as (app, pilot, fake, handle):
            mark = app.query_one(ChipMark)
            await until(pilot, lambda: mark._index >= len(frames()) - 1)
            await say(pilot, "q")                  # the first message freezes it
            await until(pilot, lambda: app.query(AssistantMessage))
    run(go())


# ---- shutting down ---------------------------------------------------------------

def _leftover_threads():
    return [t.name for t in threading.enumerate()
            if t.name.startswith(("ovat-", "fake-session")) and t.is_alive()]


def test_closing_leaves_no_task_or_thread_running():
    async def go():
        async with running() as (app, pilot, fake, handle):
            await say(pilot, "q")
            await until(pilot, lambda: app.query(AssistantMessage))
            await say(pilot, "/quit")
            await until(pilot, lambda: not app.is_running)
            await asyncio.wait_for(app.shutdown_cleanup(), 10)
            await asyncio.wait_for(handle.close(), 10)
            assert fake.closed and not handle.running
        for _ in range(100):
            if not _leftover_threads():
                break
            await asyncio.sleep(0.05)
        assert _leftover_threads() == []
    run(go())


# ---- the command table ------------------------------------------------------------

def test_every_ovat_command_is_in_the_new_apps_registry():
    from ovat.tui.agent_commands import OvatCommandRegistry
    registry = OvatCommandRegistry()
    for c in ovat_commands.COMMANDS:
        for word in (c.name, *(a for a in c.aliases if a.isalnum())):
            assert registry.parse_command(f"/{word}") is not None, word
    app_methods = dir(OvatAgentApp)
    for command in registry.commands.values():
        assert command.handler in app_methods, command.handler


def test_the_registry_keeps_a_vibe_command_only_when_the_server_serves_it():
    """The kept commands are DERIVED from the server's route table: for every
    one of Vibe's commands, the methods its handler calls must all be routes,
    or the command is not in the registry. Nothing here is a hand-kept list of
    removed commands, so it cannot drift from the server."""
    from ovat.tui.agent_commands import (CAPABILITY, VIBE_NEEDS,
                                         OvatCommandRegistry)
    from ovat.tui.server.methods import all_routes
    from ovat.tui.vibe.app_server.protocol import SERVER_METHODS
    from ovat.tui.vibe.cli.commands import CommandRegistry
    routes = set(all_routes().names())
    registry = OvatCommandRegistry()
    vibe = set(CommandRegistry().commands) | set(
        CommandRegistry(excluded_commands=[])._build_commands())
    # a command Vibe adds must be classified before it can be offered
    assert set(VIBE_NEEDS) == vibe
    for name, needs in VIBE_NEEDS.items():
        for method in needs:
            assert method.startswith(CAPABILITY) or method in SERVER_METHODS, (
                f"{name}: {method} is not a method of the protocol")
    for name in vibe:
        works = all(m in routes for m in VIBE_NEEDS[name])
        assert (name in registry.commands) == works, name
    # and the other way: nothing offered needs a method the server lacks
    for name in set(registry.commands) & vibe:
        assert all(m in routes for m in VIBE_NEEDS[name]), name
    # what that comes to today (it grows when the server learns a method)
    assert set(registry.commands) & vibe == {
        "help", "clear", "copy", "exit", "status", "theme", "resume",
        "rewind", "rename", "branch", "log", "compact", "model", "thinking",
        "reload", "config", "mcp"}


def test_a_removed_command_would_only_have_ended_in_not_supported():
    """The server answers every unrouted method with a readable error, so a
    removed command is one whose handler would hit it."""
    from ovat.tui.agent_commands import VIBE_NEEDS, OvatCommandRegistry
    from ovat.tui.server.methods import all_routes
    routes = set(all_routes().names())
    for name in VIBE_NEEDS:
        if name not in OvatCommandRegistry().commands:
            assert any(m not in routes for m in VIBE_NEEDS[name]), name


def test_ovat_agent_ui_picks_the_app(monkeypatch):
    from ovat import tui
    from ovat.tui import agent_app
    from ovat.tui import app as classic
    monkeypatch.delenv("OVAT_AGENT_UI", raising=False)
    assert tui.runner() is agent_app.run
    monkeypatch.setenv("OVAT_AGENT_UI", "classic")
    assert tui.runner() is classic.run
    monkeypatch.setenv("OVAT_AGENT_UI", "Classic ")
    assert tui.runner() is classic.run
    monkeypatch.setenv("OVAT_AGENT_UI", "anything else")
    assert tui.runner() is agent_app.run


def test_run_agent_closes_the_session_then_the_server(monkeypatch):
    """The entry point's shutdown: the app's own cleanup closes the session,
    the handle stops the server task and the agent. (The terminal side of
    `_run_app_with_cleanup` is Vibe's; here it runs the app headless.)"""
    from ovat.tui import agent_app
    made = []

    def factory(workflow, sink, cwd=None):
        fake = AppFake(workflow, sink, cwd)
        made.append(fake)
        return fake

    async def headless(app):
        async with app.run_test(size=SIZE) as pilot:
            await until(pilot, lambda: "GPU" in facts(app))
        await app.shutdown_cleanup()

    monkeypatch.setattr(agent_app, "_run_app_with_cleanup", headless)

    async def go():
        await agent_app.run_agent("w.yml", cwd=CWD, session_factory=factory)
        assert made[0].closed
        for _ in range(100):
            if not _leftover_threads():
                break
            await asyncio.sleep(0.05)
        assert _leftover_threads() == []
    run(go())


class _Engine:
    """A built agent, as ovat.agent.factory makes one."""

    def __init__(self):
        from ovat.agent.events import EventBus, observed
        self.events = EventBus()
        self.tools = {"search_docs": {"kind": "builtin",
                                      "permission": "allow"}}
        self._tool = observed("search_docs", lambda query: "x", self.events)
        self.session = Session(system_prompt="be brief")

    def run(self, text):
        self._tool(query=text)
        self.last_trace = {"engine": "native", "turns": [], "totals": {
            "prompt_tokens": 10, "completion_tokens": 5, "tool_calls": 1}}
        return "Twenty days."


def _real_session(workflow, sink, cwd=None):
    from ovat.config.workflow import WorkflowConfig
    return s.AgentSession(
        workflow, sink, cwd=cwd,
        load=lambda path: WorkflowConfig(model={"name": "m/Qwen3.5-4B-ov"}),
        build=lambda config: _Engine(), close=lambda agent: None)


def test_the_app_runs_on_the_real_agent_session_too():
    """The same app, server and client over ovat.agentic.session.AgentSession
    itself (only the model and the tools behind it are stand-ins), so the
    attributes the app reads off the session are the real ones."""
    async def go():
        async with running(session_factory=_real_session,
                           ready="Qwen3.5-4B-ov") as (
                app, pilot, session, handle):
            assert "native" in facts(app) and "w.yml" in banner_text(app)
            await say(pilot, "How many days?")
            await until(pilot, lambda: app.query(AssistantMessage))
            assert app.query(ToolGroup)
            assert "Twenty days." in app.query_one(
                AssistantMessage).get_content()
            await until(pilot, lambda: "15 tokens" in facts(app))
            await say(pilot, "/tools")
            await until(pilot, lambda: "search_docs" in screen_text(app))
            await say(pilot, "/engine react")
            await until(pilot, lambda: "react" in facts(app))
            assert session.config.agent.type == "react"
            await say(pilot, "/yolo")
            await until(pilot, lambda: session.broker.auto_approve)
            await say(pilot, "/new")
            await until(pilot, lambda: "New conversation started"
                        in screen_text(app))
            await until(pilot, lambda: handle.ready)
            await say(pilot, "again")
            await until(pilot, lambda: len(app.query(AssistantMessage)) == 1
                        and "again" in [w._content
                                        for w in app.query(UserMessage)])
    run(go())


def test_the_copied_app_builds_its_banner_through_a_hook_and_ovat_swaps_it():
    from ovat.tui.agent_app import AgentBanner
    from ovat.tui.vibe.cli.textual_ui.app import VibeApp
    from ovat.tui.vibe.cli.textual_ui.widgets.banner.banner import \
        Banner as VibeBanner
    assert VibeApp._banner_factory is VibeBanner

    async def go():
        async with running() as (app, pilot, fake, handle):
            assert isinstance(app._banner, AgentBanner)
            # (a CSS type query cannot tell the two classes called Banner apart)
            assert not any(isinstance(w, VibeBanner) for w in app.query("*"))
    run(go())


def test_reporting_an_error_without_sentry_installed_does_nothing(monkeypatch):
    """The copied app reports errors to a service OVAT does not ship. Without
    the package the call returns, instead of raising its own ImportError over
    the error it was asked to report."""
    import sys

    from ovat.tui.vibe.observability.sentry import capture_sentry_exception
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)
    assert capture_sentry_exception(RuntimeError("x"), fatal=True) is None


def test_streamed_reasoning_and_answer_deltas_grow_in_the_transcript():
    async def go():
        async with running(
                tool=False,
                reasoning_deltas=("check ", "handbook"),
                answer_deltas=("New hires ", "get 20 ", "days."),
                reasoning="<think>check handbook</think>",
                answer="New hires get 20 days.") as (app, pilot, fake, handle):
            await say(pilot, "How many days?")
            await until(pilot, lambda: app.query(AssistantMessage)
                        and "New hires get 20 days."
                        in app.query_one(AssistantMessage).get_content())
            assert len(app.query(AssistantMessage)) == 1
            assert len(app.query(ReasoningMessage)) == 1
            assert "check handbook" in app.query_one(
                ReasoningMessage)._content
    run(go())


def test_context_window_shows_percentage_in_the_bottom_bar():
    async def go():
        async with running(context_window=1000) as (app, pilot, fake, handle):
            await say(pilot, "How many days?")
            await until(pilot, lambda: "10%" in screen_text(app))
    run(go())




# ---- `ovat agent --continue / --resume` ------------------------------------------

def test_ovat_agent_continue_and_resume_on_the_command_line(
        monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from ovat.cli.main import app as cli
    from ovat.tui import app as classic
    invoke = CliRunner().invoke
    workflow = tmp_path / "w.yml"
    workflow.write_text("model: {}\n")

    result = invoke(cli, ["agent", str(workflow), "--continue",
                          "--resume", "abc"])
    assert result.exit_code == 1 and "not both" in result.output

    # nothing saved: one readable line, no traceback
    result = invoke(cli, ["agent", str(workflow), "--resume", "nope"])
    assert result.exit_code == 1, result.output
    assert "No saved session nope" in result.output
    assert "Traceback" not in result.output

    # the classic app takes neither flag, and a plain start must not pass
    # them to it (it would fail with a TypeError)
    calls = []
    monkeypatch.setattr(classic, "run", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setenv("OVAT_AGENT_UI", "classic")
    assert invoke(cli, ["agent", str(workflow)]).exit_code == 0
    assert calls == [((str(workflow),), {})]
    result = invoke(cli, ["agent", str(workflow), "--continue"])
    assert result.exit_code == 1 and "need the new app" in result.output


@pytest.mark.parametrize("cancelled", [False, True])
def test_ask_user_shows_the_question_and_answers_or_cancels(cancelled):
    from ovat.tui.vibe.cli.textual_ui.widgets.question_app import QuestionApp

    async def go():
        async with running(question={"question": "Which database?",
                                      "choices": ["sqlite", "postgres"]}) as (
                app, pilot, fake, handle):
            await say(pilot, "pick one")
            await until(pilot, lambda: app.query(QuestionApp))
            await until(pilot, lambda: "Which database?" in screen_text(app))
            question = app.query_one(QuestionApp)
            await until(pilot, lambda: not question.is_within_grace_period())
            await pilot.press("escape" if cancelled else "enter")
            await until(pilot, lambda: app.query(AssistantMessage))
            assert fake.answered == (None if cancelled else "sqlite")
            await until(pilot, lambda: not app.query(QuestionApp))
    run(go())


def test_config_edits_the_workflow_and_reloads_with_memory_kept(tmp_path):
    from ovat.tui.vibe.cli.textual_ui.screens.config.config_screen import ConfigScreen
    from ovat.config.workflow import load_workflow
    path = _wf(tmp_path)

    async def go():
        async with running(workflow=path) as (app, pilot, fake, handle):
            await say(pilot, "first")
            await until(pilot, lambda: not app._is_busy())
            await say(pilot, "/config")
            await until(pilot, lambda: isinstance(app.screen, ConfigScreen))
            await until(pilot, lambda: app.screen.query("#config-screen-options"))
            await type_text(pilot, "device")
            await pilot.press("enter")
            await until(pilot, lambda: app.screen.query("#config-edit-options"))
            await pilot.press("down", "enter")
            await until(pilot, lambda: load_workflow(path).model.device == "NPU")
            await until(pilot, lambda: isinstance(app.screen, ConfigScreen))
            await pilot.press("escape")
            await until(pilot, lambda: not isinstance(app.screen, ConfigScreen))
            await until(pilot, lambda: fake.picks == [(None, None)])
            assert [m["content"] for m in fake.memory() if m["role"] == "user"] == ["first"]
            assert "# the model" in Path(path).read_text()
    run(go())


def test_the_app_starts_with_the_saved_theme(tmp_path):
    from ovat.agentic import prefs
    prefs.update(str(tmp_path), theme="nord")

    async def go():
        async with running(cwd=str(tmp_path)) as (app, pilot, fake, handle):
            await until(pilot, lambda: app.theme == "nord")
    run(go())
