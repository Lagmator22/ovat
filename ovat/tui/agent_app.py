# ovat/tui/agent_app.py
"""`ovat agent`, on the full copied Vibe app.

    VibeApp (ovat/tui/vibe/cli/textual_ui/app.py)   the copied app
      -> speaks JSON-RPC to an app server, never to an agent
    OVAT's app server (ovat/tui/server/)            the other end of that wire
      -> drives ovat.agentic.session.AgentSession
    OvatAgentApp (this file)                        what is OVAT's in the app

The subclass is small on purpose. It adds the OVAT commands (the table in
ovat/tui/agent_commands.py, the handlers here), the OVAT banner with the chip
mark, the agent's facts and the live hardware reading in the bottom bar, and
OVAT's theme. It turns off what OVAT does not have: update checks, telemetry,
feedback prompts, the editor promo, the "what's new" of the app it was copied
from, voice and spoken summaries. Everything else is the copied app.

`/engine` and `/workflow` have no protocol method (a workflow is not part of
Vibe's app), so they call the server's handle, which rebuilds the agent and
tells the app when it is ready.
"""
from __future__ import annotations

import asyncio
import difflib
import re
import threading
from functools import partial
from pathlib import Path
from typing import Any

from textual.binding import Binding
from textual.actions import SkipAction
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Static

from ovat.agentic import session as s
from ovat.cli.theme import OVAT_THEME
from ovat.tui import agent_text
from ovat.tui.agent_commands import OvatCommandRegistry
from ovat.tui.server import OvatServerHandle, start_ovat_app_server
from ovat.tui.vibe.app_server import AppServerSession
from ovat.tui.vibe.app_server.protocol import AppServerResponseError
from ovat.tui.vibe.app_server.events import (AppServerEvent, ServerError,
                                             ServerWarning)
from ovat.tui.vibe.cli.clipboard import copy_text_to_clipboard
from ovat.tui.vibe.cli.commands import CommandRegistry
from ovat.tui.vibe.cli.textual_ui import app as vibe_app
from ovat.tui.vibe.cli.textual_ui.app import (BottomApp, StartupOptions,
                                              VibeApp, _noop_narrator_manager,
                                              _noop_voice_manager,
                                              _run_app_with_cleanup)
from ovat.tui.vibe.cli.textual_ui.count_format import format_count_markdown
from ovat.tui.vibe.cli.textual_ui.widgets.messages import (
    ErrorMessage, SlashCommandMessage, UserCommandMessage, WarningMessage)
from ovat.tui.vibe.cli.textual_ui.word_selection import WordSelectScreen
from ovat.tui.vibe.cli.textual_ui.widgets.theme_picker import (
    ThemePickerApp, sorted_theme_names)
from ovat.tui.vibe.config_values import AUTO_THEME
from ovat.tui.widgets.banner import Banner as OvatBanner

VIBE_APP_FILE = vibe_app.__file__
HISTORY_FILE = Path.home() / ".ovat" / "agent_history"
HARDWARE_EVERY_S = 2.0


class AgentBanner(OvatBanner):
    """OVAT's banner, built where the copied app builds Vibe's.

    The copied app makes its banner with Vibe's arguments (model list, MCP
    servers, skills...) and calls `set_state` with them whenever something
    changes. OVAT's banner takes those calls and ignores the arguments: what
    it shows (model, device, engine, tools, subagents) comes from
    `source()`, the live state of the agent, so it is right whenever it is
    asked, including after /engine.
    """

    def __init__(self, config: Any = None, skills_count: int = 0,
                 mcp: Any = None, *, animate: bool = True, source=None,
                 **ignored: Any) -> None:
        super().__init__(animate=animate, **{k: v for k, v in ignored.items()
                                             if k in ("id", "classes")})
        self._source = source

    def on_mount(self) -> None:
        super().on_mount()
        self.set_state(None, 0)

    def set_state(self, config: Any = None, skills_count: int = 0,
                  mcp: Any = None, **ignored: Any) -> None:
        if self._source is None or not self.is_attached:
            return
        info, problem = self._source()
        if info:
            self.show_session(info)
        elif problem:
            self.show_problem("see below")
        else:
            self.show_loading()

    def freeze_animation(self) -> None:
        self.freeze_mark()


class OvatScreen(WordSelectScreen):
    """The copied app's chat screen, with Ctrl+C as OVAT has always had it:
    with text selected it copies the selection and clears it, so the next
    Ctrl+C counts towards quitting instead of copying again. With nothing
    selected the key falls through to the app (stop the answer, or quit)."""

    def action_copy_text(self) -> None:
        if self.get_selected_text() is None:
            raise SkipAction()
        self.app.action_copy_selection()
        self.clear_selection()


class HardwareReading(Message):
    def __init__(self, reading: dict) -> None:
        super().__init__()
        self.reading = reading


class OvatAgentApp(VibeApp):
    TITLE = "OVAT agent"
    # Textual reads CSS_PATH relative to the file of the class being run, so
    # the copied sheet is named in full.
    CSS_PATH = str(Path(VIBE_APP_FILE).parent / "app.tcss")

    # The copied sheet lays the bottom bar out as path, process, spacer and
    # the context gauge; the agent's facts take the spacer's room.
    CSS = """
    #bottom-bar #spacer { display: none; }
    #bottom-bar #agent-facts {
        width: 1fr; height: auto; margin: 0 1 0 1; text-align: right;
        background: transparent; color: $text-muted;
    }
    """

    # Shift+Up/Down scroll the transcript, but not as priority bindings
    # (AGENTS.md: they stole keys from the slash menu and modals).
    BINDINGS = [
        Binding("shift+up", "scroll_chat_up", "Scroll Up", show=False),
        Binding("shift+down", "scroll_chat_down", "Scroll Down", show=False),
    ]

    def __init__(self, session: AppServerSession, handle: OvatServerHandle, *,
                 history_file: Path | None = HISTORY_FILE,
                 animate: bool = True, hardware: Any = True,
                 startup: StartupOptions | None = None,
                 **kwargs: Any) -> None:
        # No update check, no "what's new" cache, no editor promo, no voice
        # and no narrator: each is a constructor argument of the copied app.
        super().__init__(
            history_file=history_file, app_server=session,
            startup=startup or StartupOptions(autocopy_to_clipboard=False),
            update_notifier=None, update_cache_repository=None,
            voice_manager=_noop_voice_manager(),
            narrator_manager=_noop_narrator_manager(),
            vscode_extension_promo=None, **kwargs)
        self._handle = handle
        self._banner_factory = partial(AgentBanner, animate=animate,
                                       source=self._banner_source)
        self._theme_name = OVAT_THEME.name
        self.register_theme(OVAT_THEME)
        self._hardware_choice = hardware
        self._hardware = None
        self._reading: dict = {}
        self._unsubscribe = None

    # ---- the pieces the copied app asks its subclass for ----------------------

    def get_default_screen(self) -> Screen:
        return OvatScreen(id="_default")

    def _build_command_registry(self) -> CommandRegistry:
        return OvatCommandRegistry(context=self._command_context())

    def _banner_source(self) -> tuple[dict, str | None]:
        """(what the banner shows, why there is no agent). Not ready, as
        while /engine builds again, shows neither."""
        handle = self._handle
        if handle.ready:
            return handle.agent_session.info or {}, None
        return {}, handle.build_error

    @property
    def _agent(self) -> s.AgentSession:
        return self._handle.agent_session

    # ---- start, hardware, the bottom bar --------------------------------------

    def on_mount(self) -> None:
        bar = self.query_one("#bottom-bar")
        bar.mount(Static("", id="agent-facts"),
                  before=self._context_progress)
        self._unsubscribe = self._handle.subscribe(self._on_agent_report)
        if self._hardware_choice is True:
            from ovat.agentic.hardware import HardwareMonitor
            self._hardware = HardwareMonitor()
        elif self._hardware_choice:
            self._hardware = self._hardware_choice
        if self._hardware is not None:
            self.set_interval(HARDWARE_EVERY_S, self._sample_hardware)
        self._update_bar()

    def on_unmount(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._hardware is not None:
            self._hardware.close()
            self._hardware = None

    def _sample_hardware(self) -> None:
        monitor = self._hardware

        def take():
            try:
                self.post_message(HardwareReading(monitor.sample()))
            except Exception:
                pass
        threading.Thread(target=take, daemon=True, name="ovat-hw").start()

    def on_hardware_reading(self, message: HardwareReading) -> None:
        if self.is_running:
            self._reading = message.reading
            self._update_bar()

    def _update_bar(self) -> None:
        try:
            bar = self.query_one("#agent-facts", Static)
        except NoMatches:
            return
        agent = self._agent
        bar.update(agent_text.facts_text(
            agent.info or {}, agent.totals, bool(agent.serving),
            self._reading))

    def _on_agent_report(self, event: object) -> None:
        """A report from the agent session, heard through the server's
        handle (on the event loop). The banner and the bar follow it."""
        if not self.is_running:
            return
        name = type(event).__name__
        if name in ("Ready", "BuildFailed"):
            self._refresh_banner()
        if name in ("Ready", "BuildFailed", "TurnDone", "Notice"):
            self._update_bar()

    # ---- a typo in a command is not a question for the model ------------------

    async def _dispatch_submitted_value(self, raw_value: str) -> None:
        """The copied app sends an unknown `/word` to the model as a prompt,
        which spends a whole turn on a typo. Here it is said, with the
        nearest command; text with more slashes in it (a path) is a prompt."""
        value = raw_value.strip()
        word = value.split(None, 1)[0] if value else ""
        known = {a.lstrip("/") for c in self.commands.commands.values()
                 for a in c.aliases if a.startswith("/")}
        if (word.startswith("/") and re.fullmatch(r"/[\w-]+", word)
                and word[1:].lower() not in known
                and self._resolve_skill(value) is None):
            near = difflib.get_close_matches(word[1:].lower(), sorted(known),
                                             n=1)
            hint = f" Did you mean `/{near[0]}`?" if near else ""
            await self._mount_and_scroll(SlashCommandMessage(value.lstrip("/")))
            await self._note(f"No command `{word}`.{hint} `/help` lists them.",
                             "warn")
            return
        await super()._dispatch_submitted_value(raw_value)

    # ---- what the server says: in the transcript, not in toasts ---------------

    async def _handle_immediate_turn_event(self, event: AppServerEvent) -> bool:
        if isinstance(event, ServerWarning):
            warning = event.params.warning
            if warning.code == "info":
                await self._mount_and_scroll(UserCommandMessage(warning.message))
            else:
                await self._mount_and_scroll(WarningMessage(warning.message))
            return True
        if isinstance(event, ServerError):
            await self._mount_and_scroll(ErrorMessage(
                event.params.error.message, collapsed=self._tools_collapsed))
            return True
        return await super()._handle_immediate_turn_event(event)

    async def _note(self, text: str, level: str = "info") -> None:
        if level == "error":
            await self._mount_and_scroll(ErrorMessage(text))
        elif level == "warn":
            await self._mount_and_scroll(WarningMessage(text))
        else:
            await self._mount_and_scroll(UserCommandMessage(text))

    # ---- theme: OVAT's, for "auto" and as the default -------------------------

    async def _apply_theme(self, theme: str) -> None:
        """The server says "auto"; here that means the theme the person
        chose with /theme, and OVAT's own until they do. (OVAT's theme is
        not one of Textual's built-in names, which the copied code refuses.)
        A name from the picker is only a preview until it is chosen."""
        if theme == AUTO_THEME:
            theme = self._theme_name
        if theme != OVAT_THEME.name:
            await super()._apply_theme(theme)
            return
        if self.theme == OVAT_THEME.name:
            return
        before = (self.native_ansi_color, self.current_theme.dark)
        self.theme = OVAT_THEME.name
        after = (self.native_ansi_color, self.current_theme.dark)
        if after != before:
            self._restyle_diff_widgets(ansi=after[0], dark=after[1])

    async def _persist_theme(self, theme: str) -> None:
        """A theme chosen with /theme lasts for the session: writing the
        settings back comes with the config screen, in a later phase."""
        self._theme_name = theme

    async def _switch_to_theme_picker_app(self) -> None:
        if self._current_bottom_app == BottomApp.ThemePicker:
            return
        names = [OVAT_THEME.name] + [n for n in sorted_theme_names()
                                     if n != AUTO_THEME]
        await self._switch_from_input(ThemePickerApp(
            theme_names=names, current_theme=self._theme_name))

    # ---- keys -----------------------------------------------------------------

    def check_action(self, action: str, parameters: tuple[object, ...]
                     ) -> bool | None:
        """Esc belongs to a full-screen page (/telemetry) while it is open:
        the copied app's priority Esc would otherwise stop the agent behind
        it and leave the page on screen."""
        if action == "interrupt" and len(self.screen_stack) > 1:
            return False
        return super().check_action(action, parameters)

    # ---- keys of the copied app whose server side OVAT does not have ---------
    #
    # Their commands are left out of the registry, but a key reaches the same
    # code without it. Each does nothing here rather than end in "not
    # supported".

    async def _steer_queued_now(self) -> bool:
        """Ctrl+Enter, or Enter on an empty prompt, would push the waiting
        prompts into the running turn. The agent takes a new instruction only
        between turns: the queue runs them one after the other by itself."""
        return False

    async def action_toggle_debug_console(self, **kwargs: Any) -> None:
        """Ctrl+Backslash: the console reads the server's log."""

    # ---- /status: the copied one prints a cost in dollars ---------------------

    async def _show_status(self, **kwargs: Any) -> None:
        stats = self.app_server.resources.runtime.stats
        info = self._agent.info or {}
        lines = ["## Agent Statistics", ""]
        if info.get("model"):
            lines.append(f"- **Model**: {info['model']} on "
                         f"{info.get('device')}, {info.get('engine')} engine")
        lines += [
            f"- **Turns**: {format_count_markdown(stats.steps)}",
            "- **Session Prompt Tokens**: "
            f"{format_count_markdown(stats.session_prompt_tokens)}",
            "- **Session Completion Tokens**: "
            f"{format_count_markdown(stats.session_completion_tokens)}",
            "- **Session Total Tokens**: "
            f"{format_count_markdown(stats.session_total_llm_tokens)}",
            "- **Last Turn Tokens**: "
            f"{format_count_markdown(stats.last_turn_total_tokens)}"]
        message = UserCommandMessage("\n".join(lines))
        message.add_class("agent-statistics")
        await self._mount_and_scroll(message)

    # ---- OVAT's slash commands (names match ovat.agentic.commands) -----------

    async def _cmd_tools(self, **kwargs: Any) -> None:
        await self._note(agent_text.tools_markdown(self._agent.info or {}))

    async def _cmd_agents(self, **kwargs: Any) -> None:
        await self._note(agent_text.agents_markdown(
            self._agent.info or {}, bool(self._agent.serving)))

    async def _cmd_ask(self, cmd_args: str = "", **kwargs: Any) -> None:
        name, _, task = cmd_args.partition(" ")
        task = task.strip()
        if not name or not task:
            await self._note("Usage: `/ask <subagent> <task>`", "warn")
            return
        known = [x["name"] for x in (self._agent.info or {}).get(
            "subagents", [])]
        if name not in known:
            await self._note(
                f"No subagent named `{name}`. "
                + (f"This workflow has: {', '.join(known)}." if known
                   else "This workflow has no subagents."), "warn")
            return
        # The server sends "@name task" straight to that subagent.
        await self._handle_user_message(f"@{name} {task}")

    async def _cmd_serve(self, cmd_args: str = "", **kwargs: Any) -> None:
        if cmd_args.strip() == "stop":
            self._agent.stop_server()
            await self._note("Stopping the A2A server.")
        else:
            self._agent.start_server()
        self.set_timer(1.0, self._update_bar)

    async def _cmd_engine(self, cmd_args: str = "", **kwargs: Any) -> None:
        engine = cmd_args.strip()
        if engine not in agent_text.ENGINES:
            await self._note(f"Engines: {', '.join(agent_text.ENGINES)}",
                             "warn")
            return
        await self._note(f"Switching to the {engine} engine.")
        await self._rebuild(self._handle.switch_engine(engine))

    async def _cmd_workflow(self, cmd_args: str = "", **kwargs: Any) -> None:
        path = cmd_args.strip()
        if not path:
            await self._note("Usage: `/workflow path/to/workflow.yml`", "warn")
            return
        await self._rebuild(self._handle.switch_workflow(path))

    async def _rebuild(self, switching) -> None:
        """Run a switch on the server handle; the banner shows the agent is
        loading until the server reports it ready (or says why not)."""
        try:
            await switching
        except Exception as exc:
            await self._note(f"Could not switch: {exc}", "warn")
            return
        self._refresh_banner()

    async def _cmd_trace(self, **kwargs: Any) -> None:
        await self._note(agent_text.trace_markdown(self._agent.last_trace))

    async def _cmd_yolo(self, **kwargs: Any) -> None:
        """The auto-approve profile, as Shift+Tab switches it, so the
        indicator and the server agree."""
        agents = self.app_server.resources.agents
        on = agents.active.name != "auto-approve"
        await agents.switch("auto-approve" if on else "default")
        self._refresh_profile_widgets()
        await self._note(*agent_text.yolo_text(on))

    async def _cmd_save(self, cmd_args: str = "", **kwargs: Any) -> None:
        await self._note(*agent_text.save_conversation(
            self._agent.agent, self.app_server.cwd, cmd_args))

    async def _cmd_load(self, cmd_args: str = "", **kwargs: Any) -> None:
        await self._note(*agent_text.load_conversation(
            self._agent, self.app_server.cwd, cmd_args))

    async def _cmd_doctor(self, cmd_args: str = "", **kwargs: Any) -> None:
        from ovat.tui.widgets.doctor import DoctorReport
        if cmd_args.strip() == "copy":
            done = [r for r in self.query(DoctorReport)
                    if r.checks is not None]
            if not done:
                await self._note("Nothing to copy yet: run `/doctor` first.")
                return
            copy_text_to_clipboard(
                self, done[-1].report,
                success_message="Copied the doctor report")
            return
        await self._mount_and_scroll(DoctorReport(self._agent.workflow))

    async def _cmd_telemetry(self, **kwargs: Any) -> None:
        from ovat.tui.telemetry import TelemetryView
        # A callable, not the agent: /engine builds a new one.
        session = self._agent
        self.push_screen(TelemetryView(
            agent=lambda: getattr(session, "agent", None)))


# ---- the entry point -----------------------------------------------------------

class CannotOpen(Exception):
    """`--continue` / `--resume` found nothing to open; the text says why."""


async def run_agent(workflow: str | None, *, cwd: str | None = None,
                    session_factory=s.AgentSession,
                    continue_session: bool = False,
                    resume_session_id: str | None = None) -> None:
    """Start the app server, run the app, and leave nothing running.

    The app closes the session on its way out (shutdown_cleanup sends
    `session/stop`); the handle then stops the server task and the agent,
    including the A2A server process `/serve` may have started.
    """
    try:
        session, handle = await start_ovat_app_server(
            workflow, cwd=cwd, session_factory=session_factory,
            continue_session=continue_session,
            resume_session_id=resume_session_id)
    except AppServerResponseError as exc:
        # Nothing saved to continue, or no saved session with that id.
        raise CannotOpen(exc.error.message) from None
    try:
        await _run_app_with_cleanup(OvatAgentApp(session, handle))
    finally:
        await handle.close()


def run(workflow: str | None = None, *,
        continue_session: bool = False,
        resume_session_id: str | None = None) -> None:
    """Entry point for `ovat agent`."""
    asyncio.run(run_agent(workflow or s.find_workflow(),
                          continue_session=continue_session,
                          resume_session_id=resume_session_id))
