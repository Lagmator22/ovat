# ovat/tui/app.py
"""`ovat agent`: the agent TUI, built from the copied widgets in
ovat/tui/vibe/ and driven by OVAT's AgentSession.

    #chat                   ChatScroll: banner, what's new, the transcript
    #loading-area           LoadingWidget while the agent works
    #bottom-app-container   ChatInputContainer, or an ApprovalApp in its place
    #bottom-bar             PathDisplay, process, spacer, the agent's facts

The copied EventHandler draws tool calls and answers; bridge.Bridge turns
OVAT's agent events into the history events it understands. AgentSession
does the work on worker threads and reports through one callback, posted
here as a SessionEvent (post_message is thread-safe).
"""
import os
import threading
import time
from pathlib import Path

from textual.app import App
from textual.binding import Binding
from textual.containers import Horizontal, VerticalGroup
from textual.message import Message
from textual.widgets import Static

from ovat.agent.events import (REMOTE_STATUS, SUBAGENT_STATUS, TOOL_ENDED,
                               TOOL_STARTED)
from ovat.agent.permissions import ALWAYS, ONCE, REFUSE, SESSION
from ovat.agentic import commands as ovat_commands
from ovat.agentic import session as s
from ovat.cli.theme import OVAT_THEME
from ovat.tui import agent_text
from ovat.tui.bridge import Bridge
from ovat.tui.server.safe import approval_safe
from ovat.tui.vibe.agents import AgentSafety
from ovat.tui.vibe.app_server._effect_models import GenericEffectDetail
from ovat.tui.vibe.cli.commands import Command, CommandRegistry
from ovat.tui.vibe.cli.textual_ui.app import ChatScroll
from ovat.tui.vibe.cli.textual_ui.handlers.event_handler import EventHandler
from ovat.tui.vibe.cli.textual_ui.widgets.approval_app import ApprovalApp
from ovat.tui.vibe.cli.textual_ui.widgets.chat_input.container import \
    ChatInputContainer
from ovat.tui.vibe.cli.textual_ui.widgets.loading import LoadingWidget
from ovat.tui.vibe.cli.textual_ui.widgets.messages import (
    BashOutputMessage, ErrorMessage, InterruptMessage, SlashCommandMessage,
    UserCommandMessage, UserMessage, WarningMessage, WhatsNewMessage)
from ovat.tui.vibe.cli.textual_ui.widgets.no_markup_static import \
    NoMarkupStatic
from ovat.tui.vibe.cli.textual_ui.widgets.path_display import PathDisplay
from ovat.tui.vibe.utils.tool_presentation import EffectCallDisplay
from ovat.tui.widgets.banner import Banner

TCSS = Path(__file__).parent / "vibe" / "cli" / "textual_ui" / "app.tcss"
HISTORY_FILE = Path.home() / ".ovat" / "agent_history"
QUIT_CONFIRM_S = 2.0
ENGINES = agent_text.ENGINES
WHATS_NEW = (
    "**What's new in the OVAT agent**\n\n"
    "- **Subagents**: hand a task to another workflow, an A2A agent or a "
    "cloud API (`/agents`, `/ask`)\n"
    "- **A2A**: `/serve` offers this workflow to other agents\n"
    "- **Approvals**: anything that sends data off this PC asks first\n"
    "- **`!cmd`** runs a shell command, Esc stops it")


class OvatCommandRegistry(CommandRegistry):
    """The copied registry, filled from OVAT's one command table."""

    def _build_commands(self) -> dict[str, Command]:
        return {
            c.name: Command(
                aliases=frozenset({f"/{c.name}", *(f"/{a}" for a in c.aliases
                                                   if a.isalnum())}),
                description=c.help + (f"  ({c.usage})" if c.usage else ""),
                handler=f"_cmd_{c.name}")
            for c in ovat_commands.COMMANDS}


class SessionEvent(Message):
    def __init__(self, event) -> None:
        super().__init__()
        self.event = event


class HardwareReading(Message):
    def __init__(self, reading: dict) -> None:
        super().__init__()
        self.reading = reading


class OvatAgentApp(App):
    TITLE = "OVAT agent"
    CSS_PATH = str(TCSS)

    BINDINGS = [
        Binding("escape", "interrupt", show=False),
        Binding("ctrl+c", "cancel_or_quit", show=False, priority=True),
        Binding("ctrl+o", "toggle_tools", show=False),
    ]

    def __init__(self, workflow: str | None = None, *,
                 session_factory=s.AgentSession, animate: bool = True,
                 history_file: Path | None = HISTORY_FILE,
                 cwd: str | None = None, hardware=True) -> None:
        super().__init__()
        self._workflow = workflow
        self._session_factory = session_factory
        self._animate_mark = animate
        self._history_file = history_file
        self._cwd = cwd or os.getcwd()
        self.session = None
        self._chat_screen = None
        self._bridge = Bridge()
        self._handler: EventHandler | None = None
        self._loading: LoadingWidget | None = None
        self._bash: BashOutputMessage | None = None
        self._approvals: list = []
        self._approval: ApprovalApp | None = None
        self._approval_request = None
        self._last_quit_press = 0.0
        self._hardware_choice = hardware
        self._hardware = None
        self._reading: dict = {}
        self._tools_collapsed = True

    # ---- layout --------------------------------------------------------------

    def compose(self):
        with ChatScroll(id="chat"):
            yield Banner(animate=self._animate_mark, id="banner")
            yield WhatsNewMessage(WHATS_NEW)
            yield VerticalGroup(id="messages")
        with Horizontal(id="loading-area"):
            yield Static(id="loading-area-content")
        with Static(id="bottom-app-container"):
            yield ChatInputContainer(
                command_registry=OvatCommandRegistry(),
                history_file=self._history_file, id="input-container")
        with Horizontal(id="bottom-bar"):
            yield PathDisplay(self._cwd)
            yield NoMarkupStatic(f"[PID {os.getpid()}]", id="process-title")
            yield NoMarkupStatic(id="spacer")
            yield Static("", id="agent-facts")

    def on_mount(self) -> None:
        self._chat_screen = self.screen
        self.register_theme(OVAT_THEME)
        self.theme = OVAT_THEME.name
        self._handler = EventHandler(
            mount_callback=self._mount_message,
            get_tools_collapsed=lambda: self._tools_collapsed)
        self.session = self._session_factory(
            self._workflow, self._from_session, cwd=self._cwd)
        self.session.start()
        self._chat.anchor()
        self._input.focus_input()
        self._update_bar()
        if self._hardware_choice is True:
            from ovat.agentic.hardware import HardwareMonitor
            self._hardware = HardwareMonitor()
        elif self._hardware_choice:
            self._hardware = self._hardware_choice
        if self._hardware is not None:
            self.set_interval(2.0, self._sample_hardware)

    def on_unmount(self) -> None:
        if self.session is not None:
            self.session.close()
        if self._hardware is not None:
            self._hardware.close()

    # ---- helpers -------------------------------------------------------------

    @property
    def _chat(self) -> ChatScroll:
        return self._chat_screen.query_one("#chat", ChatScroll)

    @property
    def _input(self) -> ChatInputContainer:
        return self._chat_screen.query_one(ChatInputContainer)

    @property
    def _messages(self) -> VerticalGroup:
        return self._chat_screen.query_one("#messages", VerticalGroup)

    async def _mount_message(self, widget, after=None, before=None, *,
                             container=None) -> None:
        """The copied app's _mount_and_scroll (Vibe app.py), without its
        message queue and history pruning, which OVAT does not have yet."""
        from ovat.tui.vibe.cli.textual_ui.widgets.messages import (
            StreamingMessageBase)
        chat = self._chat
        user_initiated = isinstance(widget, (UserMessage, UserCommandMessage))
        should_anchor = user_initiated or chat.is_at_bottom
        before_parent = before.parent if before is not None else None
        after_parent = after.parent if after is not None else None
        with self.batch_update():
            if before_parent is not None:
                await before_parent.mount(widget, before=before)
            elif after_parent is not None:
                await after_parent.mount(widget, after=after)
            elif container is not None:
                await container.mount(widget)
            else:
                await self._messages.mount(widget)
            if not widget.is_mounted:
                return
            if isinstance(widget, StreamingMessageBase):
                await widget.write_initial_content()
        if should_anchor:
            chat.anchor()

    async def note(self, text: str, level: str = "info") -> None:
        if level == "error":
            await self._mount_message(ErrorMessage(text))
        elif level == "warn":
            await self._mount_message(WarningMessage(text))
        else:
            await self._mount_message(UserCommandMessage(text))

    async def _start_loading(self, status: str | None = None) -> None:
        await self._stop_loading()
        self._loading = LoadingWidget(status)
        await self._chat_screen.query_one("#loading-area-content").mount(
            self._loading)

    async def _stop_loading(self) -> None:
        if self._loading is not None:
            await self._loading.remove()
            self._loading = None

    def _sample_hardware(self) -> None:
        monitor = self._hardware

        def take():
            try:
                self.post_message(HardwareReading(monitor.sample()))
            except Exception:
                pass
        threading.Thread(target=take, daemon=True, name="ovat-hw").start()

    def on_hardware_reading(self, message: HardwareReading) -> None:
        if self.is_running and self._chat_screen.query("#agent-facts"):
            self._reading = message.reading
            self._update_bar()

    def _update_bar(self) -> None:
        facts = agent_text.facts_text(
            self.session.info if self.session else {},
            self.session.totals if self.session else {},
            bool(self.session and self.session.serving), self._reading)
        self._chat_screen.query_one("#agent-facts", Static).update(facts)

    def _from_session(self, event) -> None:
        self.post_message(SessionEvent(event))

    # ---- session reports -----------------------------------------------------

    async def on_session_event(self, message: SessionEvent) -> None:
        if not self.is_running or not self._chat_screen.query("#agent-facts"):
            return            # a late report while the app closes
        handler = getattr(self, f"_on_{type(message.event).__name__}", None)
        if handler is not None:
            await handler(message.event)

    async def _on_Ready(self, event: s.Ready) -> None:
        self._chat_screen.query_one(Banner).show_session(event.info)
        self._input.set_agent_name(event.info.get("engine", ""))
        self._update_bar()

    async def _on_BuildFailed(self, event: s.BuildFailed) -> None:
        self._chat_screen.query_one(Banner).show_problem("see below")
        await self.note(event.text, "error")

    async def _on_TurnStarted(self, event: s.TurnStarted) -> None:
        await self._start_loading()

    async def _on_AgentActivity(self, event: s.AgentActivity) -> None:
        ev = event.event
        if ev.type not in (TOOL_STARTED, TOOL_ENDED, SUBAGENT_STATUS,
                           REMOTE_STATUS):
            return
        kinds = {t["name"]: t["kind"]
                 for t in (self.session.info or {}).get("tools", [])}
        history = self._bridge.convert(ev, kinds)
        if history is not None:
            await self._handler.handle_event(history, self._loading)

    async def _on_ApprovalNeeded(self, event: s.ApprovalNeeded) -> None:
        self._approvals.append(event.request)
        if self._approval is None:
            await self._show_next_approval()

    async def _on_QuestionNeeded(self, event: s.QuestionNeeded) -> None:
        # This older screen has no way to show a question, so it is "no
        # answer" at once: the model decides itself instead of waiting.
        self.session.questions.resolve(event.request.id, None)

    async def _show_next_approval(self) -> None:
        request = self._approvals.pop(0)
        self._approval_request = request
        if self._loading is not None:
            self._loading.begin_action_required("Waiting for approval")
        detail = GenericEffectDetail(
            tool_name=request.tool,
            input=approval_safe(request.arguments) or None,
            display=EffectCallDisplay(summary=request.tool,
                                      status_text=f"Asking about "
                                                  f"{request.tool}"))
        self._approval = ApprovalApp(effect=detail, config=None,
                                     reason=request.reason or None)
        # As the copied app's _switch_from_input: hide AND disable the input,
        # or its text area pulls the focus back and the keys meant for the
        # panel go into a hidden prompt.
        with self.batch_update():
            self._input.display = False
            self._input.disabled = True
            await self._chat_screen.query_one("#bottom-app-container").mount(
                self._approval)
        self.call_after_refresh(self._approval.focus)

    async def _answer_approval(self, decision: str) -> None:
        if self._approval_request is not None:
            self.session.broker.resolve(self._approval_request.id, decision)
        self._approval_request = None
        if self._approval is not None:
            await self._approval.remove()
            self._approval = None
        if self._loading is not None:
            self._loading.end_action_required()
        if self._approvals:
            await self._show_next_approval()
            return
        self._input.disabled = False
        self._input.display = True
        self.call_after_refresh(self._input.focus_input)

    async def on_approval_app_approval_granted(self, message) -> None:
        await self._answer_approval(ONCE)

    async def on_approval_app_approval_granted_always_tool(self, message) -> None:
        await self._answer_approval(SESSION)

    async def on_approval_app_approval_granted_always_permanent(
            self, message) -> None:
        await self._answer_approval(ALWAYS)

    async def on_approval_app_approval_rejected(self, message) -> None:
        await self._answer_approval(REFUSE)

    async def _on_TurnDone(self, event: s.TurnDone) -> None:
        await self._stop_loading()
        if self._handler.current_tool_group is not None:
            self._handler._finalize_tool_group(escalate=event.failed)
        if event.cancelled:
            await self._mount_message(InterruptMessage())
        elif event.failed:
            await self._mount_message(ErrorMessage(event.answer or "failed"))
        else:
            await self._handler.handle_event(
                self._bridge.answer(event.answer or "(no answer)"))
        await self._handler.finalize_streaming()
        self._approvals.clear()
        if self._approval is not None:
            await self._answer_approval(REFUSE)
        self._update_bar()

    async def _on_ShellLine(self, event: s.ShellLine) -> None:
        if self._bash is not None:
            await self._bash.append_output(event.text + "\n")
            self.call_after_refresh(self._chat.scroll_end, animate=False)

    async def _on_ShellDone(self, event: s.ShellDone) -> None:
        if self._bash is not None:
            interrupted = event.code is None or event.code < 0
            await self._bash.finish(event.code if event.code is not None
                                    else -1, interrupted=interrupted)
            self._bash = None

    async def _on_Notice(self, event: s.Notice) -> None:
        await self.note(event.text, event.level)
        self._update_bar()

    # ---- input ---------------------------------------------------------------

    async def on_chat_input_container_submitted(
            self, message: ChatInputContainer.Submitted) -> None:
        text = message.value.strip()
        if not text:
            return
        if text.startswith("!"):
            await self._run_shell(text[1:].strip())
            return
        if text.startswith("/"):
            await self._slash(text)
            return
        if self.session.busy:
            self._input.value = text          # give it back, do not lose it
            await self.note("Still working on the last one; Esc stops it.",
                            "warn")
            return
        await self._mount_message(UserMessage(text))
        self.session.ask(text)

    async def _run_shell(self, command: str) -> None:
        if not command:
            await self.note("Type a command after !, like !dir or !ls", "warn")
            return
        self._bash = BashOutputMessage(command, self._cwd, pending=True)
        await self._mount_message(self._bash)
        if self.session.run_shell(command) is None:
            self._bash = None

    async def _slash(self, text: str) -> None:
        await self._mount_message(SlashCommandMessage(text))
        command, rest = ovat_commands.parse(text)
        if command is None:
            near = ovat_commands.complete(rest or "")
            hint = f" Did you mean `/{near[0].name}`?" if near and rest else ""
            await self.note(f"No command `/{rest}`.{hint} `/help` lists them.",
                            "warn")
            return
        await getattr(self, f"_cmd_{command.name}")(rest)

    # ---- slash commands (names match ovat.agentic.commands.COMMANDS) --------

    async def _cmd_help(self, rest: str) -> None:
        lines = ["### Commands", "", *agent_text.commands_markdown()]
        lines += ["", "### Keys", "",
                  "- `Enter` send, `Ctrl+J` new line, `Up/Down` history",
                  "- `!cmd` runs a shell command here",
                  "- `Esc` stops the answer or the command",
                  "- `Ctrl+O` opens or folds the tool calls",
                  "- `Ctrl+C` copies a selection; twice quits"]
        await self.note("\n".join(lines))

    async def _cmd_new(self, rest: str) -> None:
        await self._messages.remove_children()
        self.session.new_conversation()

    async def _cmd_tools(self, rest: str) -> None:
        await self.note(agent_text.tools_markdown(self.session.info))

    async def _cmd_agents(self, rest: str) -> None:
        await self.note(agent_text.agents_markdown(
            self.session.info, self.session.serving))

    async def _cmd_ask(self, rest: str) -> None:
        name, _, task = rest.partition(" ")
        if not name or not task.strip():
            await self.note("Usage: `/ask <subagent> <task>`", "warn")
            return
        self.session.ask_subagent(name, task.strip())

    async def _cmd_serve(self, rest: str) -> None:
        if rest.strip() == "stop":
            self.session.stop_server()
            await self.note("Stopping the A2A server.")
        else:
            self.session.start_server()
        self.set_timer(1.0, self._update_bar)

    async def _cmd_engine(self, rest: str) -> None:
        engine = rest.strip()
        if engine not in ENGINES:
            await self.note(f"Engines: {', '.join(ENGINES)}", "warn")
            return
        await self.note(f"Switching to the {engine} engine.")
        self.session.start(engine)

    async def _cmd_trace(self, rest: str) -> None:
        await self.note(agent_text.trace_markdown(self.session.last_trace))

    async def _cmd_yolo(self, rest: str) -> None:
        broker = self.session.broker
        broker.auto_approve = not broker.auto_approve
        self._input.set_safety(AgentSafety.YOLO if broker.auto_approve
                               else AgentSafety.NEUTRAL)
        await self.note(*agent_text.yolo_text(broker.auto_approve))

    async def _cmd_workflow(self, rest: str) -> None:
        path = rest.strip()
        if not path:
            await self.note("Usage: `/workflow path/to/workflow.yml`", "warn")
            return
        self.session.switch_workflow(path)

    async def _cmd_save(self, rest: str) -> None:
        await self.note(*agent_text.save_conversation(
            self.session.agent, self._cwd, rest))

    async def _cmd_load(self, rest: str) -> None:
        await self.note(*agent_text.load_conversation(
            self.session, self._cwd, rest))

    async def _cmd_copy(self, rest: str) -> None:
        if not self.session.last_answer:
            await self.note("Nothing to copy yet.")
            return
        self.copy_to_clipboard(self.session.last_answer)
        self.notify("Copied the last answer.", timeout=2)

    async def _cmd_doctor(self, rest: str) -> None:
        from ovat.tui.widgets.doctor import DoctorReport
        if rest.strip() == "copy":
            done = [r for r in self._chat_screen.query(DoctorReport)
                    if r.checks is not None]
            if not done:
                await self.note("Nothing to copy yet: run `/doctor` first.")
                return
            self.copy_to_clipboard(done[-1].report)
            self.notify("Copied the doctor report.", timeout=2)
            return
        await self._mount_message(DoctorReport(self.session.workflow))

    async def _cmd_telemetry(self, rest: str) -> None:
        from ovat.tui.telemetry import TelemetryView
        # A callable, not the agent: /engine builds a new one.
        session = self.session
        self.push_screen(TelemetryView(
            agent=(lambda: getattr(session, "agent", None))
            if session is not None else None))

    async def _cmd_quit(self, rest: str) -> None:
        self.exit()

    # ---- keys ----------------------------------------------------------------

    def action_interrupt(self) -> None:
        if self._input.dismiss_completion():
            return
        if self.session is not None and self.session.cancel():
            if self._loading is not None:
                self._loading.set_status("Interrupting")

    def action_toggle_tools(self) -> None:
        from ovat.tui.vibe.cli.textual_ui.widgets.tools import ToolGroup
        self._tools_collapsed = not self._tools_collapsed
        for group in self._chat_screen.query(ToolGroup):
            group.set_collapsed(self._tools_collapsed)
        for section in self._chat_screen.query(".doctor-ok"):
            section.set_collapsed(self._tools_collapsed)

    def action_cancel_or_quit(self) -> None:
        if self.screen.get_selected_text():
            self.screen.action_copy_text()
            self.screen.clear_selection()
            self.notify("Copied the selection.", timeout=2)
            return
        if self.session is not None and self.session.cancel():
            return
        now = time.monotonic()
        if now - self._last_quit_press < QUIT_CONFIRM_S:
            self.exit()
            return
        self._last_quit_press = now
        self.notify("Press Ctrl+C again to quit", severity="warning",
                    timeout=QUIT_CONFIRM_S)


def run(workflow: str | None = None) -> None:
    """Entry point for `ovat agent`."""
    OvatAgentApp(workflow or s.find_workflow()).run()
