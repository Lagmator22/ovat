# ovat/tui/app.py
"""`ovat agent`: the chat-first agent TUI.

Layout, top to bottom (Mistral Vibe's shape, which gives the conversation
every row it can):

    #chat          scrolls: the banner, then the transcript
    #loading-area  one line while the agent works (spinner, status, time)
    #bottom        the input box, or an approval panel in its place
    #bottom-bar    folder, workflow, model on device, tokens, flags

The app only draws. AgentSession (ovat/agentic/session.py) does the work on
worker threads and reports through one callback; that callback posts a
SessionEvent here (post_message is thread-safe), and on_session_event
turns each report into widgets. So nothing in this file blocks, and every
rule worth testing is tested without a terminal.
"""
import os
import time
from pathlib import Path

from rich.text import Text
from textual.app import App
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static

from ovat.agent.events import (REMOTE_STATUS, SUBAGENT_STATUS, TOOL_ENDED,
                               TOOL_STARTED)
from ovat.agentic import commands
from ovat.agentic import session as s
from ovat.cli import ui
from ovat.cli.theme import OVAT_THEME
from ovat.tui.widgets.approval import ApprovalPanel
from ovat.tui.widgets.banner import Banner
from ovat.tui.widgets.input import HISTORY_FILE, InputBox
from ovat.tui.widgets.loading import LoadingLine
from ovat.tui.widgets.messages import (AssistantMessage, Notice, ShellBlock,
                                      UserMessage, turn_footer)
from ovat.tui.widgets.tools import ToolGroup

QUIT_CONFIRM_S = 2.0
ENGINES = ("native", "react", "llamaindex", "openai-agents")


class HardwareReading(Message):
    """A CPU / NPU / RAM reading, taken on a worker thread."""

    def __init__(self, reading: dict) -> None:
        super().__init__()
        self.reading = reading


class SessionEvent(Message):
    """One report from the AgentSession, carried to the UI thread."""

    def __init__(self, event) -> None:
        super().__init__()
        self.event = event


def _short_path(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home):] if path.startswith(home) else path


def help_text() -> Text:
    out = Text()
    out.append("Commands\n", style=f"bold {ui.CYAN}")
    width = max(len(c.name) + len(c.usage) for c in commands.COMMANDS) + 3
    for c in commands.COMMANDS:
        label = f"/{c.name}" + (f" {c.usage}" if c.usage else "")
        out.append(f"  {label.ljust(width)}", style=f"bold {ui.PURPLE}")
        out.append(c.help + "\n", style=ui.DIM)
    out.append("\nKeys\n", style=f"bold {ui.CYAN}")
    for key, what in (("enter", "send"), ("ctrl+j", "new line"),
                      ("!cmd", "run a shell command here"),
                      ("esc", "stop the answer or the command"),
                      ("ctrl+o", "open or fold the tool calls"),
                      ("up/down", "earlier messages"),
                      ("ctrl+c", "copy a selection; twice to quit")):
        out.append(f"  {key.ljust(width)}", style=f"bold {ui.PURPLE}")
        out.append(what + "\n", style=ui.DIM)
    return out


class OvatAgentApp(App):
    TITLE = "OVAT agent"

    CSS = """
    Screen { background: $background; layout: vertical; }
    #chat { height: 1fr; width: 100%; padding: 0 1; scrollbar-size: 1 1; }
    #messages { height: auto; width: 100%; }
    #loading-area { height: auto; width: 100%; padding: 0 1; }
    #bottom { height: auto; width: 100%; padding: 0 1; }
    #bottom-bar { height: 1; width: 100%; padding: 0 1;
                  color: $text-muted; }
    """

    BINDINGS = [
        Binding("escape", "interrupt", show=False),
        # priority so it also wins over Textual's own ctrl+c -> copy_text;
        # action_cancel_or_quit copies a selection itself, first.
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
        self._play_intro = animate
        self._history_file = history_file
        self._cwd = cwd or os.getcwd()
        self.session = None
        self._loading: LoadingLine | None = None
        self._group: ToolGroup | None = None
        self._shell_block: ShellBlock | None = None
        self._approvals: list = []
        self._panel: ApprovalPanel | None = None
        self._last_quit_press = 0.0
        self._turn_started = 0.0
        # True: the real monitor; False: none; anything else: a fake (tests).
        self._hardware_choice = hardware
        self._hardware = None
        self._reading: dict = {}

    # ---- layout ------------------------------------------------------------

    def compose(self):
        with VerticalScroll(id="chat"):
            yield Banner(animate=self._play_intro, id="banner")
            yield Vertical(id="messages")
        yield Vertical(id="loading-area")
        with Vertical(id="bottom"):
            yield InputBox(history_file=self._history_file, id="input-box")
        yield Static("", id="bottom-bar")

    def on_mount(self) -> None:
        self.register_theme(OVAT_THEME)
        self.theme = OVAT_THEME.name
        self.session = self._session_factory(
            self._workflow, self._from_session, cwd=self._cwd)
        self.session.start()
        self._update_bar()
        self.query_one(InputBox).area.focus()
        if self._hardware_choice is True:
            from ovat.agentic.hardware import HardwareMonitor
            self._hardware = HardwareMonitor()
        elif self._hardware_choice:
            self._hardware = self._hardware_choice
        if self._hardware is not None:
            self.set_interval(2.0, self._sample_hardware)

    def _sample_hardware(self) -> None:
        monitor = self._hardware

        def take():
            try:
                self.post_message(HardwareReading(monitor.sample()))
            except Exception:
                pass
        import threading
        threading.Thread(target=take, daemon=True,
                         name="ovat-hardware").start()

    def on_hardware_reading(self, message: HardwareReading) -> None:
        self._reading = message.reading
        self._update_bar()

    def on_unmount(self) -> None:
        if self.session is not None:
            self.session.close()
        if self._hardware is not None:
            self._hardware.close()

    def _from_session(self, event) -> None:
        """Called on worker threads. post_message is the thread-safe door."""
        self.post_message(SessionEvent(event))

    # ---- helpers -----------------------------------------------------------

    @property
    def _messages(self) -> Vertical:
        return self.query_one("#messages", Vertical)

    def _add(self, widget) -> None:
        self._messages.mount(widget)
        self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

    def note(self, text, level: str = "info") -> None:
        self._add(Notice(text, level))

    def _start_loading(self, status: str = "Thinking") -> None:
        self._stop_loading()
        self._loading = LoadingLine(status)
        self.query_one("#loading-area").mount(self._loading)

    def _stop_loading(self) -> None:
        if self._loading is not None:
            self._loading.remove()
            self._loading = None

    def _update_bar(self) -> None:
        info = self.session.info if self.session else {}
        bar = Text(no_wrap=True, overflow="ellipsis")
        bar.append(_short_path(self._cwd), style=ui.DIM)
        if info.get("name"):
            bar.append("  ·  ", style=ui.DIM)
            bar.append(info["name"], style=ui.PURPLE)
        if info.get("model"):
            bar.append("  ·  ", style=ui.DIM)
            bar.append(f"{info['device']}", style=f"bold {ui.CYAN}")
            bar.append(f" {info['model']}", style=ui.DIM)
            bar.append(f"  ·  {info.get('engine')}", style=ui.DIM)
        totals = self.session.totals if self.session else {}
        used = totals.get("prompt_tokens", 0) + totals.get(
            "completion_tokens", 0)
        if used:
            bar.append(f"  ·  {used:,} tokens", style=ui.DIM)
        if self.session and self.session.serving:
            bar.append("  ·  ", style=ui.DIM)
            bar.append("a2a serving", style=ui.GREEN)
        if self.session and self.session.broker.auto_approve:
            bar.append("  ·  ", style=ui.DIM)
            bar.append("yolo", style=f"bold {ui.YELLOW}")
        if self._reading:
            from ovat.agentic.hardware import format_reading
            bar.append("  ·  ", style=ui.DIM)
            bar.append(format_reading(self._reading), style=ui.CYAN)
        self.query_one("#bottom-bar", Static).update(bar)

    # ---- what the session reports -------------------------------------------

    def on_session_event(self, message: SessionEvent) -> None:
        event = message.event
        handler = getattr(self, f"_on_{type(event).__name__}", None)
        if handler is not None:
            handler(event)

    def _on_Ready(self, event: s.Ready) -> None:
        self.query_one(Banner).show_session(event.info)
        engine = event.info.get("engine", "")
        self.query_one(InputBox).set_title(f"{engine}")
        self._update_bar()

    def _on_BuildFailed(self, event: s.BuildFailed) -> None:
        self.query_one(Banner).show_problem("see below")
        self.note(event.text, "error")
        self._update_bar()

    def _on_TurnStarted(self, event: s.TurnStarted) -> None:
        self._group = None
        self._turn_started = time.monotonic()
        self._start_loading("Thinking")

    def _on_AgentActivity(self, event: s.AgentActivity) -> None:
        ev = event.event
        if ev.type not in (TOOL_STARTED, TOOL_ENDED, REMOTE_STATUS,
                           SUBAGENT_STATUS):
            return
        if self._group is None:
            kinds = {t["name"]: t["kind"]
                     for t in self.session.info.get("tools", [])}
            self._group = ToolGroup(kinds)
            self._add(self._group)
        self._group.handle(ev)
        if self._loading is not None:
            d = ev.data
            if ev.type == TOOL_STARTED:
                args = d.get("arguments") or {}
                who = args.get("agent") if d.get("name") == "delegate" else None
                self._loading.set_status(
                    f"Delegating to {who}" if who
                    else f"Calling {d.get('name')}")
            elif ev.type == TOOL_ENDED:
                self._loading.set_status("Thinking")
            else:
                self._loading.set_status(f"{d.get('agent')} working")

    def _on_ApprovalNeeded(self, event: s.ApprovalNeeded) -> None:
        self._approvals.append(event.request)
        if self._panel is None:
            self._show_next_approval()

    def _show_next_approval(self) -> None:
        if not self._approvals:
            return
        request = self._approvals.pop(0)
        if self._loading is not None:
            self._loading.pause()
            self._loading.set_status("Waiting for your approval")
        self.query_one(InputBox).display = False
        self._panel = ApprovalPanel(request)
        self.query_one("#bottom").mount(self._panel)
        self._panel.focus()

    def on_approval_panel_decided(self, message: ApprovalPanel.Decided) -> None:
        self.session.broker.resolve(message.request_id, message.decision)
        if self._panel is not None:
            self._panel.remove()
            self._panel = None
        if self._loading is not None:
            self._loading.resume()
            self._loading.set_status("Thinking")
        if self._approvals:
            self._show_next_approval()
            return
        box = self.query_one(InputBox)
        box.display = True
        box.area.focus()

    def _on_TurnDone(self, event: s.TurnDone) -> None:
        self._stop_loading()
        if self._group is not None:
            self._group.finish(event.failed and not event.cancelled,
                               event.elapsed)
        footer = turn_footer(event.elapsed, event.totals, event.sources,
                             event.failed, event.cancelled)
        if event.cancelled:
            self._add(AssistantMessage("_Stopped._", footer))
        else:
            self._add(AssistantMessage(event.answer or "_(no answer)_",
                                       footer, failed=event.failed))
        self._group = None
        self._clear_approvals()
        self._update_bar()

    def _on_ShellLine(self, event: s.ShellLine) -> None:
        if self._shell_block is not None:
            self._shell_block.add_line(event.text)
            self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

    def _on_ShellDone(self, event: s.ShellDone) -> None:
        if self._shell_block is not None:
            self._shell_block.finish(event.code, event.elapsed)
            self._shell_block = None
        if not (self.session and self.session.busy):
            self._stop_loading()

    def _on_Notice(self, event: s.Notice) -> None:
        self.note(event.text, event.level)
        self._update_bar()

    def _clear_approvals(self) -> None:
        self._approvals.clear()
        if self._panel is not None:
            self._panel.remove()
            self._panel = None
            box = self.query_one(InputBox)
            box.display = True
            box.area.focus()

    # ---- what the person typed ----------------------------------------------

    def on_input_box_submitted(self, message: InputBox.Submitted) -> None:
        text = message.text
        if text.startswith("!"):
            self._run_shell(text[1:].strip())
            return
        if text.startswith("/"):
            self._slash(text)
            return
        if self.session.busy:
            # Give the question back instead of losing it.
            self.query_one(InputBox).set_text(text)
            self.note("Still working on the last one; esc stops it.", "warn")
            return
        self._add(UserMessage(text))
        self.session.ask(text)

    def _run_shell(self, command: str) -> None:
        if not command:
            self.note("Type a command after !, like !dir or !ls", "warn")
            return
        self._shell_block = ShellBlock(command)    # its header shows the command
        self._add(self._shell_block)
        if not self.session.busy:
            self._start_loading(f"Running {command.split()[0]}")
        if self.session.run_shell(command) is None:
            self._shell_block = None

    def _slash(self, text: str) -> None:
        command, rest = commands.parse(text)
        if command is None:
            near = commands.complete(rest or "")
            hint = f" Did you mean /{near[0].name}?" if near and rest else ""
            self.note(f"No command /{rest}.{hint} /help lists them.", "warn")
            return
        getattr(self, f"_cmd_{command.name}")(rest)

    # ---- slash commands (names match commands.COMMANDS) ---------------------

    def _cmd_help(self, rest: str) -> None:
        self.note(help_text())

    def _cmd_new(self, rest: str) -> None:
        self._messages.remove_children()
        self.session.new_conversation()

    def _cmd_tools(self, rest: str) -> None:
        tools = self.session.info.get("tools", [])
        if not tools:
            self.note("This agent has no tools.")
            return
        out = Text()
        for t in tools:
            colour = {"allow": ui.GREEN, "ask": ui.YELLOW,
                      "deny": ui.RED}.get(t["permission"], ui.DIM)
            out.append(f"{t['name']:<20}", style="bold")
            out.append(f"{t['kind']:<12}", style=ui.DIM)
            out.append(t["permission"] + "\n", style=colour)
        self.note(out)

    def _cmd_agents(self, rest: str) -> None:
        info = self.session.info
        out = Text()
        subs = info.get("subagents", [])
        out.append("Subagents\n", style=f"bold {ui.CYAN}")
        if not subs:
            out.append("  none (add a subagents: list to the workflow)\n",
                       style=ui.DIM)
        for sub in subs:
            out.append(f"  {sub['name']:<16}", style=f"bold {ui.PURPLE}")
            out.append(f"{sub['where']:<10}", style=ui.DIM)
            out.append(sub["description"] + "\n")
        peers = [t for t in info.get("tools", []) if t["kind"] == "a2a"]
        out.append("A2A agents used as tools\n", style=f"bold {ui.CYAN}")
        if not peers:
            out.append("  none\n", style=ui.DIM)
        for peer in peers:
            out.append(f"  {peer['name']:<16}", style="bold")
            out.append(peer["permission"] + "\n", style=ui.DIM)
        out.append("This workflow over A2A\n", style=f"bold {ui.CYAN}")
        state = ("serving (`/serve stop` ends it)" if self.session.serving
                 else "not serving (`/serve` starts it)")
        out.append(f"  {state}", style=ui.DIM)
        self.note(out)

    def _cmd_ask(self, rest: str) -> None:
        name, _, task = rest.partition(" ")
        if not name or not task.strip():
            self.note("Usage: /ask <subagent> <task>", "warn")
            return
        self._add(UserMessage(f"@{name} {task}"))
        self.session.ask_subagent(name, task.strip())

    def _cmd_serve(self, rest: str) -> None:
        if rest.strip() == "stop":
            self.session.stop_server()
            self.note("Stopping the A2A server.")
        else:
            self.session.start_server()
        self.set_timer(1.0, self._update_bar)

    def _cmd_engine(self, rest: str) -> None:
        engine = rest.strip()
        if engine not in ENGINES:
            self.note(f"Engines: {', '.join(ENGINES)}", "warn")
            return
        self.query_one(Banner).display = True
        self.note(f"Switching to the {engine} engine…")
        self.session.start(engine)

    def _cmd_trace(self, rest: str) -> None:
        trace = self.session.last_trace
        if not trace:
            self.note("Nothing yet: ask something first.")
            return
        totals = trace.get("totals", {})
        out = Text()
        out.append(f"{trace.get('engine')}  ", style=f"bold {ui.CYAN}")
        flags = [k for k in ("failed", "truncated", "undecoded_tool_call",
                             "empty_answer") if totals.get(k)]
        out.append(f"{totals.get('turns')} turns, "
                   f"{totals.get('tool_calls')} tool calls, "
                   f"{totals.get('latency_s')} s", style=ui.DIM)
        if flags:
            out.append("  " + ", ".join(flags), style=ui.RED)
        for i, turn in enumerate(trace.get("turns", []) or [], 1):
            out.append(f"\n  turn {i}: {turn.get('latency_s')} s, "
                       f"{turn.get('prompt_tokens')} → "
                       f"{turn.get('completion_tokens')} tokens, "
                       f"{turn.get('finish_reason')}", style=ui.DIM)
            for call in turn.get("tool_calls", []):
                out.append(f"\n    {call.get('name')} "
                           f"{call.get('duration_s')} s", style=ui.DIM)
        self.note(out)

    def _cmd_yolo(self, rest: str) -> None:
        broker = self.session.broker
        broker.auto_approve = not broker.auto_approve
        self.note("Every tool is approved for this session. /yolo again "
                  "turns it off." if broker.auto_approve else
                  "Tools ask again.", "warn" if broker.auto_approve else "info")
        self._update_bar()

    def _cmd_workflow(self, rest: str) -> None:
        path = rest.strip()
        if not path:
            self.note("Usage: /workflow path/to/workflow.yml", "warn")
            return
        self.session.switch_workflow(path)
        self.query_one(Banner).display = True

    def _sessions_dir(self) -> Path:
        return Path(self._cwd) / ".ovat" / "sessions"

    def _cmd_save(self, rest: str) -> None:
        memory = getattr(self.session.agent, "session", None)
        if memory is None:
            self.note("Saving works on the native engine.", "warn")
            return
        name = rest.strip() or time.strftime("agent-%Y%m%d-%H%M%S")
        path = self._sessions_dir() / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        memory.save(str(path))
        self.note(f"Saved to {_short_path(str(path))}")

    def _cmd_load(self, rest: str) -> None:
        from ovat.agent.session import Session
        name = rest.strip()
        path = self._sessions_dir() / f"{name}.json"
        if not name or not path.is_file():
            saved = sorted(p.stem for p in self._sessions_dir().glob("*.json"))
            self.note("Usage: /load <name>. Saved: "
                      + (", ".join(saved[-8:]) or "none"), "warn")
            return
        if getattr(self.session.agent, "session", None) is None:
            self.note("Loading works on the native engine.", "warn")
            return
        self.session.agent.session = Session.load(str(path))
        turns = sum(1 for m in self.session.agent.session.messages
                    if m.get("role") == "user")
        self.note(f"Loaded {name} ({turns} questions). The model remembers "
                  f"it; the transcript above is not redrawn.")

    def _cmd_copy(self, rest: str) -> None:
        if not self.session.last_answer:
            self.note("Nothing to copy yet.")
            return
        self.copy_to_clipboard(self.session.last_answer)
        self.notify("Copied the last answer.", timeout=2)

    def _cmd_doctor(self, rest: str) -> None:
        from ovat.cli.doctor_screen import DoctorScreen
        self.push_screen(DoctorScreen(self.session.workflow))

    def _cmd_telemetry(self, rest: str) -> None:
        from ovat.cli.telemetry_screen import TelemetryScreen
        self.push_screen(TelemetryScreen())

    def _cmd_quit(self, rest: str) -> None:
        self.exit()

    # ---- keys ----------------------------------------------------------------

    def action_interrupt(self) -> None:
        if self.session is not None and self.session.cancel():
            if self._loading is not None:
                self._loading.set_status("Interrupting")

    def action_toggle_tools(self) -> None:
        groups = list(self.query(ToolGroup))
        if groups:
            groups[-1].toggle()

    def action_cancel_or_quit(self) -> None:
        """Selection: copy it. Busy: stop. Idle: quit on a SECOND press."""
        if self.screen.get_selected_text():
            self.screen.action_copy_text()
            self.screen.clear_selection()
            self.notify("Copied the selection.", timeout=2)
            return
        area = self.query_one(InputBox).area
        if area.has_focus and area.selected_text:
            self.copy_to_clipboard(area.selected_text)
            self.notify("Copied.", timeout=2)
            return
        if self.session is not None and self.session.cancel():
            return
        now = time.monotonic()
        if now - self._last_quit_press < QUIT_CONFIRM_S:
            self.exit()
            return
        self._last_quit_press = now
        self.notify("Press Ctrl-C again to quit  ·  /quit works too",
                    severity="warning", timeout=QUIT_CONFIRM_S)


def run(workflow: str | None = None) -> None:
    """Entry point for `ovat agent`."""
    OvatAgentApp(workflow or s.find_workflow()).run()
