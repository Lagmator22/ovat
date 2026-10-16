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

from rich.text import Text
from textual.app import App
from textual.binding import Binding
from textual.containers import Horizontal, VerticalGroup
from textual.message import Message
from textual.widgets import Static

from ovat.agent.events import (REMOTE_STATUS, SUBAGENT_STATUS, TOOL_ENDED,
                               TOOL_STARTED)
from ovat.agent.permissions import ONCE, REFUSE, SESSION
from ovat.agentic import commands as ovat_commands
from ovat.agentic import session as s
from ovat.cli import ui
from ovat.cli.theme import OVAT_THEME
from ovat.tui.bridge import Bridge
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
ENGINES = ("native", "react", "llamaindex", "openai-agents")
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


_APPROVAL_ARG_MAX = 600


def approval_safe(value):
    """A model-written argument as shown in the approval panel.

    The person deciding must see what will be sent, but the text must not
    be able to dress up as part of the panel: a "task" containing
    "\n1. Allow once" would otherwise print lines that look like the
    panel's own options, or push the real ones out of view. So control
    characters go, every newline shows as a visible mark on ONE line, and
    long text is cut (with the length said)."""
    if isinstance(value, dict):
        return {str(k): approval_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [approval_safe(v) for v in value]
    if not isinstance(value, str):
        return value
    text = "".join(ch if ch.isprintable() or ch == "\n" else "?"
                   for ch in value)
    text = text.replace("\r\n", "\n").replace("\n", " \u21b5 ")
    if len(text) > _APPROVAL_ARG_MAX:
        text = (text[:_APPROVAL_ARG_MAX]
                + f" ... [{len(value)} characters in all]")
    return text


def _short_path(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home):] if path.startswith(home) else path


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
        info = self.session.info if self.session else {}
        facts = Text(no_wrap=True, overflow="ellipsis")
        if info.get("model"):
            facts.append(info["device"], style=f"bold {ui.CYAN}")
            facts.append(f" {info['model']} · {info.get('engine')}")
        totals = self.session.totals if self.session else {}
        used = totals.get("prompt_tokens", 0) + totals.get(
            "completion_tokens", 0)
        if used:
            facts.append(f" · {used:,} tokens")
        if self.session and self.session.serving:
            facts.append(" · ")
            facts.append("a2a serving", style=ui.GREEN)
        if self._reading:
            from ovat.agentic.hardware import format_reading
            facts.append(" · " + format_reading(self._reading).replace(
                "  ·  ", " · "))
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
        await self._answer_approval(SESSION)

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
        lines = ["### Commands", ""]
        for c in ovat_commands.COMMANDS:
            usage = f" {c.usage}" if c.usage else ""
            lines.append(f"- `/{c.name}{usage}`: {c.help}")
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
        tools = self.session.info.get("tools", [])
        if not tools:
            await self.note("This agent has no tools.")
            return
        lines = ["| tool | kind | permission |", "| --- | --- | --- |"]
        lines += [f"| `{t['name']}` | {t['kind']} | {t['permission']} |"
                  for t in tools]
        await self.note("\n".join(lines))

    async def _cmd_agents(self, rest: str) -> None:
        info = self.session.info
        lines = ["### Subagents", ""]
        subs = info.get("subagents", [])
        lines += ([f"- `{x['name']}` ({x['where']}): {x['description']}"
                   for x in subs] or ["- none (add `subagents:` to the "
                                      "workflow)"])
        peers = [t for t in info.get("tools", []) if t["kind"] == "a2a"]
        lines += ["", "### A2A agents used as tools", ""]
        lines += ([f"- `{p['name']}` ({p['permission']})" for p in peers]
                  or ["- none"])
        lines += ["", "### This workflow over A2A", "",
                  "- serving (`/serve stop` ends it)" if self.session.serving
                  else "- not serving (`/serve` starts it)"]
        await self.note("\n".join(lines))

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
        trace = self.session.last_trace
        if not trace:
            await self.note("Nothing yet: ask something first.")
            return
        totals = trace.get("totals", {})
        flags = [k for k in ("failed", "truncated", "undecoded_tool_call",
                             "empty_answer") if totals.get(k)]
        lines = [f"**{trace.get('engine')}**: {totals.get('turns')} turns, "
                 f"{totals.get('tool_calls')} tool calls, "
                 f"{totals.get('latency_s')} s"
                 + (f", **{', '.join(flags)}**" if flags else ""), ""]
        for i, turn in enumerate(trace.get("turns", []) or [], 1):
            lines.append(f"- turn {i}: {turn.get('latency_s')} s, "
                         f"{turn.get('prompt_tokens')} → "
                         f"{turn.get('completion_tokens')} tokens, "
                         f"{turn.get('finish_reason')}")
            for call in turn.get("tool_calls", []):
                lines.append(f"  - `{call.get('name')}` "
                             f"{call.get('duration_s')} s")
        await self.note("\n".join(lines))

    async def _cmd_yolo(self, rest: str) -> None:
        broker = self.session.broker
        broker.auto_approve = not broker.auto_approve
        self._input.set_safety(AgentSafety.YOLO if broker.auto_approve
                               else AgentSafety.NEUTRAL)
        await self.note("Every tool is approved for this session. `/yolo` "
                        "again turns it off." if broker.auto_approve
                        else "Tools ask again.",
                        "warn" if broker.auto_approve else "info")

    async def _cmd_workflow(self, rest: str) -> None:
        path = rest.strip()
        if not path:
            await self.note("Usage: `/workflow path/to/workflow.yml`", "warn")
            return
        self.session.switch_workflow(path)

    def _sessions_dir(self) -> Path:
        return Path(self._cwd) / ".ovat" / "sessions"

    async def _cmd_save(self, rest: str) -> None:
        memory = getattr(self.session.agent, "session", None)
        if memory is None:
            await self.note("Saving works on the native engine.", "warn")
            return
        name = rest.strip() or time.strftime("agent-%Y%m%d-%H%M%S")
        path = self._sessions_dir() / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        memory.save(str(path))
        await self.note(f"Saved to `{_short_path(str(path))}`")

    async def _cmd_load(self, rest: str) -> None:
        from ovat.agent.session import Session
        name = rest.strip()
        path = self._sessions_dir() / f"{name}.json"
        if not name or not path.is_file():
            saved = sorted(p.stem for p in self._sessions_dir().glob("*.json"))
            await self.note("Usage: `/load <name>`. Saved: "
                            + (", ".join(saved[-8:]) or "none"), "warn")
            return
        if getattr(self.session.agent, "session", None) is None:
            await self.note("Loading works on the native engine.", "warn")
            return
        self.session.agent.session = Session.load(str(path))
        await self.note(f"Loaded `{name}`. The model remembers it; the "
                        f"transcript above is not redrawn.")

    async def _cmd_copy(self, rest: str) -> None:
        if not self.session.last_answer:
            await self.note("Nothing to copy yet.")
            return
        self.copy_to_clipboard(self.session.last_answer)
        self.notify("Copied the last answer.", timeout=2)

    async def _cmd_doctor(self, rest: str) -> None:
        from ovat.cli.doctor_screen import DoctorScreen
        self.push_screen(DoctorScreen(self.session.workflow))

    async def _cmd_telemetry(self, rest: str) -> None:
        from ovat.cli.telemetry_screen import TelemetryScreen
        self.push_screen(TelemetryScreen())

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
