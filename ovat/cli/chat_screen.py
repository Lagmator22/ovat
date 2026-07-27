# ovat/cli/chat_screen.py
"""The native chat screen: talk to your indexed documents inside the TUI.

Unlike every other TUI command (which runs `ovat ...` as a subprocess), chat
runs IN-PROCESS. That is what makes it feel like a chat app: the model loads
once and stays warm, replies stream token by token, and the conversation has
real memory (prior turns go back into the prompt via rag_chat's history).

Layout: a scrolling transcript of message WIDGETS (Prompt and Response, both
markdown), a slash-command menu, and an input. Each answer streams into its
own Response widget, which is anchored to the bottom of the scroll view so a
long answer keeps itself in sight. That is what retired the old separate
streaming line, along with the tail-clipping and max-height it needed to stop
it shoving the input off the screen.

Esc stops a generation in flight, or leaves the screen when idle. /save and
/load persist the conversation as JSON under .ovat/sessions/.

Every colour is a Textual theme variable, so the whole conversation follows
the theme picked from the command palette; ovat/cli/theme.py makes OVAT's own
palette the default one.

The heavy lifting (config → retriever + local LLM) sits behind the
_build_components seam so tests can swap in fakes and drive the whole screen
headless in milliseconds.
"""
import json
import os
import re
import time

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.color import Color
from textual.screen import ModalScreen, Screen
from textual.containers import Vertical, VerticalScroll
from textual.widgets import (Collapsible, Footer, Input, Label, Markdown,
                             OptionList, Static)
from textual.widgets.option_list import Option

from ovat.agent.rag_chat import rag_chat
from ovat.agent.session import Session
from ovat.cli import ui
from ovat.cli.commands import ScreenCommands
from ovat.cli.editing import InputHistory
from ovat.cli.widgets import ChatInput

# The slash menu for this screen. Unlike the doctor screen's, choosing one
# INSERTS it rather than running it: most of these take an argument
# (/tokens 2048, /save demo, /copy all), so the useful next step is typing,
# not executing. Enter on the bare command still runs it.
CHAT_COMMANDS = [
    ("/engine", "swap between local  and  ovms (tools)"),
    ("/thinking", "show or hide the model's reasoning"),
    ("/copy", "copy the last answer  (or 'me' / 'all')"),
    ("/tokens", "answer length cap  (0 = no cap)"),
    ("/save", "save this conversation"),
    ("/load", "pick from your saved conversations"),
    ("/back", "return to the launcher"),
]

# Both live under .ovat/ in the directory the user launched from, so a
# project's chats and preferences stay with that project.
# 256 cut real answers off mid-sentence often enough that /tokens existed
# mainly to undo it. 1024 is a working default for both engines; /tokens is
# still there for the long ones.
DEFAULT_MAX_TOKENS = 1024

# How many of the most recent answers are rebuilt as real Markdown when a
# conversation is loaded. Every Markdown widget explodes into a child widget
# per block, per table row, per list item: measured, thirty turns of typical
# answer built 3,158 widgets and blocked the UI thread for 2.8 seconds on a
# fast machine, which is the "it loads then hangs" you feel. Older turns are
# rebuilt as plain text instead, so the whole conversation is still there and
# still readable; only the formatting of the far scrollback is given up.
REPLAY_RICH_TURNS = 6

PREFS_FILE = "chat_prefs.json"
SESSIONS_DIR = "sessions"

# Reasoning models narrate before they answer. Qwen3 and the DeepSeek-R1
# family wrap that narration in <think>…</think>; some exports spell it
# <thinking>. Both spellings, any case, across newlines.
_THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>",
                          re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Drop the reasoning blocks, leaving the answer.

    Interesting once, clutter every time after: a single Qwen3 answer can
    spend fifteen lines deciding what the question meant before saying
    anything. This affects the DISPLAY only. The session keeps the raw text,
    so /thinking on can bring it back and /save never loses it.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    # An unclosed block means generation stopped mid-thought: Esc, or the
    # token cap ran out. Everything from the opening tag on is reasoning, so
    # it goes too, rather than leaving a dangling "<think>" in the transcript.
    match = _THINK_OPEN.search(cleaned)
    if match:
        cleaned = cleaned[:match.start()]
    return cleaned.strip()


def split_thinking(text: str) -> tuple:
    """(reasoning, answer). Reasoning is "" when the model did not narrate.

    The two halves want different rendering. An answer is markdown and should
    be rendered as such; reasoning is wrapped in <think> tags, which markdown
    reads as HTML and silently swallows, so showing it verbatim is the only
    way /thinking on actually shows anything.
    """
    reasoning = [match.group(0) for match in _THINK_BLOCK.finditer(text)]
    remainder = _THINK_BLOCK.sub("", text)
    unclosed = _THINK_OPEN.search(remainder)
    if unclosed:
        reasoning.append(remainder[unclosed.start():])
    return "\n".join(reasoning).strip(), strip_thinking(text)


def _ovat_dir(cwd: str) -> str:
    return os.path.join(cwd, ".ovat")


def load_prefs(cwd: str) -> dict:
    """The last config/model-path used, so `/chat` alone just works next time."""
    try:
        with open(os.path.join(_ovat_dir(cwd), PREFS_FILE), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_prefs(cwd: str, config_path: str, model_path: str) -> None:
    os.makedirs(_ovat_dir(cwd), exist_ok=True)
    with open(os.path.join(_ovat_dir(cwd), PREFS_FILE), "w", encoding="utf-8") as f:
        json.dump({"config": config_path, "model_path": model_path}, f, indent=2)


def _session_path(cwd: str, name: str) -> str:
    # A session name is a bare word; the path and extension are ours.
    safe = "".join(c for c in name if c.isalnum() or c in "-_") or "last"
    return os.path.join(_ovat_dir(cwd), SESSIONS_DIR, f"{safe}.json")


def list_sessions(cwd: str) -> list:
    """Every saved conversation, newest first.

    Returns (name, path, modified, turns) so the picker can show more than a
    filename: which conversation this was, and how much of it there is.
    """
    folder = os.path.join(_ovat_dir(cwd), SESSIONS_DIR)
    sessions = []
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    for filename in names:
        if not filename.endswith(".json"):
            continue
        path = os.path.join(folder, filename)
        try:
            with open(path, encoding="utf-8") as handle:
                messages = json.load(handle)
            turns = sum(1 for m in messages if m.get("role") == "user")
            first = next((m["content"] for m in messages
                          if m.get("role") == "user" and m.get("content")), "")
        except (OSError, ValueError, TypeError):
            # A half-written or hand-edited file should not hide every other
            # session from the list.
            turns, first = 0, "(unreadable)"
        sessions.append({
            "name": filename[:-len(".json")],
            "path": path,
            "modified": os.path.getmtime(path),
            "turns": turns,
            "first": first,
        })
    return sorted(sessions, key=lambda s: s["modified"], reverse=True)


class SessionPicker(ModalScreen[str]):
    """Pick a saved conversation to load.

    A modal rather than another dropdown: this is a decision that pauses the
    conversation, and ModalScreen dims what is behind it to say so. It
    dismisses with the chosen name, or nothing at all if you press Esc, which
    keeps the choosing here and the loading in the chat screen.
    """

    DEFAULT_CSS = """
    SessionPicker {
        align: center middle;
    }
    #picker-frame {
        width: 70%;
        max-width: 80;
        height: auto;
        max-height: 80%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    #picker-title { color: $accent; text-style: bold; padding-bottom: 1; }
    #picker-hint { color: $text-muted; padding-top: 1; }
    #picker-list { height: auto; max-height: 14; background: $surface; }
    #picker-list > .option-list--option-highlighted {
        background: $primary;
        color: $foreground;
        text-style: bold;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=True)]

    def __init__(self, sessions: list):
        super().__init__()
        self._sessions = sessions

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-frame"):
            yield Label(f"Saved conversations ({len(self._sessions)})",
                        id="picker-title")
            yield OptionList(*[Option(_session_label(s), id=s["name"])
                               for s in self._sessions], id="picker-list")
            yield Label("Enter loads  ·  Esc cancels", id="picker-hint")

    def on_mount(self) -> None:
        self.query_one("#picker-list", OptionList).focus()
        # A short fade so the dialog arrives rather than snapping into place.
        # 120ms: long enough to read as motion, short enough not to be a wait.
        self.styles.opacity = 0.0
        self.styles.animate("opacity", value=1.0, duration=0.12)

    def action_cancel(self) -> None:
        self.dismiss("")

    def on_option_list_option_selected(self,
                                       event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id or "")


def _session_label(session: dict) -> Text:
    """One row: the name, when it was last touched, and how it opened."""
    when = time.strftime("%d %b %H:%M", time.localtime(session["modified"]))
    label = Text()
    label.append(f"{session['name']:<14} ", style=f"bold {ui.CYAN}")
    label.append(f"{when}  ", style=ui.DIM)
    label.append(f"{session['turns']} turn"
                 f"{'' if session['turns'] == 1 else 's'}  ", style=ui.DIM)
    opening = session["first"].replace("\n", " ")
    label.append(opening[:38] + ("…" if len(opening) > 38 else ""),
                 style=ui.PURPLE)
    return label


def _build_components(config_path: str, model_path: str,
                      max_tokens: int = DEFAULT_MAX_TOKENS):
    """config + model path -> (cfg, retriever, llm). The heavy step.

    Module-level on purpose: it is the seam tests monkeypatch to drive the
    screen with fakes instead of loading a real OpenVINO model.
    """
    from ovat.agent.factory import build_rag
    from ovat.config.workflow import load_workflow
    from ovat.core.model_scout import identify_model, pick_chat_llm
    from ovat.providers.llm_genai import GenAILLMProvider

    cfg = load_workflow(config_path)
    if cfg.rag is None:
        raise ValueError(
            "this workflow has no rag: section; add one and run `ovat index` first"
        )
    # Identify BEFORE the 30-second load. A vision/whisper/embedding folder
    # used to load "fine" and then explode at generate time with a C++
    # traceback about tensor ports; now it is one readable sentence.
    kind, why = identify_model(model_path)
    if kind not in ("llm", "unknown"):
        _, llms = pick_chat_llm()
        suggestion = (" Try: " + ", ".join(m["name"] for m in llms)
                      if llms else "")
        raise ValueError(
            f"{os.path.basename(model_path.rstrip(os.sep))} is not a text "
            f"LLM ({why}); chat needs a text model.{suggestion}"
        )
    retriever = build_rag(cfg)
    llm = GenAILLMProvider(model_path, device="CPU", max_new_tokens=max_tokens)
    return cfg, retriever, llm


class LocalEngine:
    """Retrieve, then answer with an openvino_genai model loaded in-process.

    This is what chat has always done, and it reads only the `rag:` half of
    the workflow. The `model:` half (ovms_url, tool_parser, name) belongs to
    the OVMS path and is ignored here, which is exactly the thing that makes
    the config confusing: an OpenVINO model folder is loadable directly by
    openvino_genai, so no server is involved and no tool calling is possible.
    """

    name = "local"
    footnote_label = "sources"

    def __init__(self, cfg, retriever, llm):
        self.cfg, self.retriever, self.llm = cfg, retriever, llm

    @property
    def max_new_tokens(self):
        return self.llm.max_new_tokens

    @max_new_tokens.setter
    def max_new_tokens(self, value):
        self.llm.max_new_tokens = value

    def describe(self) -> str:
        return "local openvino_genai · retrieval only, no tools"

    def ask(self, question: str, history: list, on_token):
        return rag_chat(self.retriever, self.llm, question, top_k=4,
                        system_prompt=self.cfg.agent.system_prompt,
                        history=history, on_token=on_token)

    def close(self) -> None:
        if self.retriever is not None:
            self.retriever.close()


class OVMSEngine:
    """The full agent loop against OVMS: real tool calling.

    Uses the `model:` section the local engine ignores, so this is the path
    where ovms_url and tool_parser finally mean something. Two honest
    differences from local, both surfaced to the user rather than hidden:
    there is no token streaming (OVMSLLMProvider answers in one shot), and
    conversation memory lives in the agent's own Session, so /load restores
    the transcript on screen but not what the agent remembers.
    """

    name = "ovms"
    footnote_label = "tools used"

    def __init__(self, cfg, agent, max_tokens=None):
        self.cfg, self.agent = cfg, agent
        if max_tokens is not None:
            self.max_new_tokens = max_tokens

    @property
    def max_new_tokens(self):
        llm = getattr(self.agent, "llm", None)      # react has no .llm
        return getattr(llm, "max_tokens", None)

    @max_new_tokens.setter
    def max_new_tokens(self, value):
        llm = getattr(self.agent, "llm", None)
        if llm is not None:
            llm.max_tokens = value

    def describe(self) -> str:
        tools = ", ".join(self.agent.tools) or "none declared"
        return (f"OVMS at {self.cfg.model.ovms_url} · "
                f"{self.cfg.model.name} · tools: {tools}")

    def ask(self, question: str, history: list, on_token):
        # history is ignored on purpose: AgentLoop keeps its own Session, and
        # replaying ours on top would duplicate every turn in the prompt.
        answer = self.agent.run(question)
        trace = getattr(self.agent, "last_trace", {}) or {}
        used = [call["name"] for turn in trace.get("turns", [])
                for call in turn.get("tool_calls", [])]
        return answer, list(dict.fromkeys(used))     # de-duped, order kept

    def close(self) -> None:
        pass


def _build_engine(config_path: str, model_path: str, engine: str = "local",
                  max_tokens: int = DEFAULT_MAX_TOKENS):
    """Build the engine `/engine` selects. The seam tests monkeypatch."""
    if engine == "ovms":
        from ovat.agent.factory import build_agent
        from ovat.config.workflow import load_workflow

        cfg = load_workflow(config_path)
        # Constructing the OVMS client does NOT connect, so a server that is
        # down fails on the first question with a readable error rather than
        # refusing to open the screen.
        return OVMSEngine(cfg, build_agent(cfg), max_tokens)
    return LocalEngine(*_build_components(config_path, model_path, max_tokens))


class ChatMessage(Markdown):
    """One turn, rendered as markdown, remembering its own source text.

    Textual's Markdown widget keeps its parsed tree, not the text it came
    from. The transcript needs the source back for /thinking redraws and for
    tests, so this holds on to it.
    """

    def __init__(self, text: str = "") -> None:
        # open_links=False is a safety decision, not a style one. Textual's
        # default handler calls app.open_url() on click, and these widgets
        # render MODEL output: a single click would launch a browser at
        # whatever href the model wrote, hallucinated or otherwise. On the AI
        # PC clicking one took the whole TUI down with it. The screen handles
        # LinkClicked itself instead.
        super().__init__(text, open_links=False)
        self.text = text

    def update(self, markdown: str):
        self.text = markdown
        return super().update(markdown)


class Prompt(ChatMessage):
    """Something the user asked."""


class Response(ChatMessage):
    """One answer, streamed into place as it generates."""

    BORDER_TITLE = "ovat"


class TranscriptLine(Static):
    """A plain line in the transcript that remembers its own text.

    Textual 8 moved Static's content behind the Content API, so there is no
    attribute to read the text back from. The transcript needs that for
    /thinking redraws and for tests, so it is kept here.

    Colour comes from a CSS class rather than an inline style, which is what
    lets these follow the active theme like everything else on the screen.
    """

    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(text, **kwargs)
        self.text = text


class Reasoning(Collapsible):
    """A model's <think> narration, folded away behind a title.

    /thinking on used to dump fifteen lines of narration above every answer,
    which solved "I cannot see it" by creating "I cannot see past it". A
    Collapsible keeps it one keypress away and out of the way until asked
    for, so showing reasoning no longer costs you the conversation.

    The body is a Static, not a Markdown widget: markdown reads <think> as
    an HTML tag and drops the whole block, so rendering it would show
    nothing at all.
    """

    def __init__(self, text: str) -> None:
        lines = text.count("\n") + 1
        super().__init__(
            Static(text, classes="reasoning-body"),
            title=f"reasoning · {lines} line{'' if lines == 1 else 's'}",
            collapsed=True,
        )
        # The transcript reads this back for /thinking redraws and tests.
        self.text = text


class Sources(TranscriptLine):
    """The citation line under an answer."""


class PlainAnswer(TranscriptLine):
    """An older answer, rebuilt as text rather than rendered markdown.

    Same content, one widget instead of the hundred a Markdown widget builds
    for a long answer with tables and lists. Used for everything past
    REPLAY_RICH_TURNS when a conversation is loaded, which is what keeps a
    long history from freezing the app while it rebuilds.
    """


class Note(TranscriptLine):
    """A status line: confirmations, refusals, errors."""


class Thinking(Static):
    """A pulsing bar shown while the model is generating.

    A static "thinking…" tells you the app is alive; a moving one tells you
    it is still alive, which is the question you actually have thirty
    seconds into a CPU generation. It breathes between two oranges rather
    than blinking, because a blink reads as an alarm and this is not one.

    The animation reschedules itself from on_complete instead of running on
    a timer, so it stops the moment the widget is removed and never leaves a
    callback firing at a dead widget.
    """

    PALE = "#FFB86C"
    DEEP = "#C2410C"
    FPS = 8            # frames a second; every one is a repaint over SSH
    SPEED = 0.10       # fraction of the cycle per frame: ~2.5s there and back

    def on_mount(self) -> None:
        self._phase = 0.0
        self.styles.color = Color.parse(self.PALE)
        # A timer, NOT styles.animate with an on_complete chain. That was the
        # obvious way and it silently did nothing: the animation never fired
        # at mount time, so the word sat at one colour forever. Measured
        # before changing it, six samples over two seconds, all identical.
        # set_interval also stops itself when the widget is removed, which
        # the self-rescheduling version had to be told to do.
        self.set_interval(1 / self.FPS, self._step)

    def _step(self) -> None:
        self._phase = (self._phase + self.SPEED) % 2.0
        # Ping-pong 0..1..0 so it breathes rather than snapping back.
        factor = self._phase if self._phase <= 1.0 else 2.0 - self._phase
        self.styles.color = Color.parse(self.PALE).blend(
            Color.parse(self.DEEP), factor)


class ChatCommands(ScreenCommands):
    """Palette entries that only exist while a chat is open.

    These duplicate the slash commands ON PURPOSE. The slash menu is for
    someone already typing; the palette is for someone who does not yet know
    the slash menu exists, which is most people the first time.
    """

    def commands(self) -> list:
        screen = self.screen
        showing = screen._show_thinking
        return [
            (f"{'Hide' if showing else 'Show'} reasoning",
             "the <think> narration a reasoning model writes before answering",
             lambda: screen.run_worker(screen._toggle_thinking(""))),
            ("Copy the last answer", "put it on the clipboard",
             lambda: screen._copy("")),
            ("Copy the whole conversation", "every turn, with roles",
             lambda: screen._copy("all")),
            ("Save this conversation", "write it to .ovat/sessions/last.json",
             lambda: screen.run_worker(screen._slash("/save"))),
        ]


class ChatScreen(Screen):
    """One conversation with the indexed documents, streaming and stateful."""

    # Screen-scoped, so these appear in the palette here and nowhere else.
    COMMANDS = {ChatCommands}

    # Every colour here is a THEME variable, never a hex literal. ovat/cli/
    # theme.py registers OVAT's palette as the default Textual theme, so these
    # come out in brand colours; pick another theme from the command palette
    # and the whole conversation follows it, which hardcoded hex could not do.
    DEFAULT_CSS = """
    ChatScreen { background: $background; }
    #chat-header { padding: 0 2; height: auto; }
    #chat-view {
        height: 1fr;
        border: round $primary;
        background: $surface;
        padding: 0 1;
        margin: 1 2 0 2;
    }
    /* The user, indented from the right so the two speakers are legible at a
       glance without reading a word of them. */
    /* Hover feedback: the transcript is selectable and copyable, so a row
       that lights under the pointer tells you it is a thing, not a print. */
    Prompt:hover { background: $primary 25%; }
    Response:hover { background: $success 14%; }
    Prompt {
        background: $primary 15%;
        color: $foreground;
        margin: 1 8 0 1;
        padding: 0 2;
    }
    /* The assistant, framed and indented from the left. */
    Response {
        border: wide $success;
        background: $success 8%;
        color: $foreground;
        margin: 1 1 0 8;
        padding: 0 2;
    }
    Reasoning {
        margin: 1 1 0 8;
        background: $surface;
        border: none;
    }
    Reasoning:hover { background: $boost; }
    .reasoning-body { color: $text-muted; padding: 0 1; }
    Sources { color: $text-muted; margin: 0 1 0 8; padding: 0 2; }
    PlainAnswer {
        margin: 1 1 0 8;
        padding: 0 2;
        color: $foreground 85%;
        border-left: outer $success 50%;
    }
    Note { color: $text-muted; margin: 0 1 0 2; padding: 0 2; }
    Thinking { margin: 0 1 0 8; padding: 0 2; text-style: italic; }
    Note.ok { color: $success; }
    Note.warn { color: $warning; }
    Note.error { color: $error; }
    #chat-palette {
        display: none;
        height: auto;
        max-height: 7;
        margin: 0 2;
        border: round $secondary;
        background: $surface;
    }
    #chat-palette > .option-list--option-highlighted {
        background: $primary;
        color: $foreground;
        text-style: bold;
    }
    /* Grows with the prompt and stops before it owns the screen; past
       eight lines it scrolls instead. */
    #chat-input {
        margin: 0 2 1 2;
        border: round $secondary;
        height: auto;
        max-height: 8;
    }
    #chat-input:focus { border: round $accent; }
    """

    BINDINGS = [
        Binding("escape", "back", show=False),
    ]

    def __init__(self, config_path: str, model_path: str, cwd: str | None = None):
        super().__init__()
        self._config_path = config_path
        self._model_path = model_path
        self._cwd = cwd or os.getcwd()
        self._engine = None              # LocalEngine or OVMSEngine once loaded
        self._session = Session()        # the transcript; system stays in rag_chat
        self._busy = False
        self._stop_stream = False        # Esc mid-generation sets this
        self._show_thinking = False      # /thinking; hidden is the useful default
        self._engine_name = "local"      # /engine; ovms adds tool calling
        self._history = InputHistory()   # Up/Down recall, like a shell
        # Its own autosave slot, so conversations accumulate instead of
        # overwriting each other. Every chat used to autosave to "last",
        # which meant the picker could only ever offer the newest one and
        # every earlier conversation was silently gone. The name is fixed at
        # open, so switching engines mid-conversation keeps writing to the
        # same file rather than forking the history.
        self._autosave = time.strftime("auto-%Y%m%d-%H%M%S")

    def compose(self) -> ComposeResult:
        header = Text()
        header.append("OVAT chat", style=f"bold {ui.CYAN}")
        header.append(f"  ·  {os.path.basename(self._config_path)}"
                      f"  ·  {os.path.basename(self._model_path)}"
                      f"  ·  engine: {self._engine_name}", style=ui.DIM)
        yield Static(header, id="chat-header")
        transcript = VerticalScroll(id="chat-view")
        transcript.border_title = "conversation"
        transcript.border_subtitle = "Ctrl-P for commands"
        yield transcript
        yield OptionList(id="chat-palette")
        yield ChatInput(placeholder="loading the model…", id="chat-input",
                       soft_wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat-input", ChatInput).focus()
        view = self.query_one("#chat-view", VerticalScroll)
        # Textual's animated overlay on the transcript is fine HERE, unlike on
        # the old append-only log: nothing has been written yet, so there is
        # no content for it to hide.
        view.loading = True
        self._load_components()

    # ---- helpers ----------------------------------------------------------

    @property
    def _view(self) -> VerticalScroll:
        return self.query_one("#chat-view", VerticalScroll)

    def _note(self, message: str, role: str = "") -> None:
        """A one-line status entry: confirmation, refusal, or error.

        `role` is a CSS class (ok / warn / error), not a colour, so these
        follow the active theme instead of pinning a hex value.
        """
        self._view.mount(Note(message, classes=role))
        self._view.scroll_end(animate=False)

    def _refresh_header(self) -> None:
        header = Text()
        header.append("OVAT chat", style=f"bold {ui.CYAN}")
        header.append(f"  ·  {os.path.basename(self._config_path)}"
                      f"  ·  {os.path.basename(self._model_path)}"
                      f"  ·  engine: {self._engine_name}", style=ui.DIM)
        self.query_one("#chat-header", Static).update(header)

    def _set_placeholder(self, text: str) -> None:
        self.query_one("#chat-input", ChatInput).placeholder = text

    def _stop_loading(self) -> None:
        self._view.loading = False

    def _mark_idle(self) -> None:
        self._busy = False
        self._stop_stream = False
        self._set_placeholder("ask about your docs  ·  / for commands  ·  "
                              "Shift-Enter for a new line")

    def _for_display(self, text: str) -> str:
        """One place decides whether reasoning is on screen or not."""
        return text if self._show_thinking else strip_thinking(text)

    async def _replay(self) -> None:
        """Rebuild the transcript from the session.

        The session always holds the RAW text, so this is what makes
        /thinking mean what it says: hiding reasoning removes it from turns
        already on screen, not just from the next one. /load reuses it.
        """
        view = self._view
        await view.remove_children()
        # Build the whole transcript first, then mount it in ONE call.
        # Awaiting a mount per widget meant a layout pass per widget, and
        # every Response is a Markdown widget that parses and mounts children
        # of its own, so loading a twenty-turn conversation was forty
        # sequential layouts on the UI thread and the app sat there frozen.
        widgets, answers = [], []
        for message in self._session.messages:
            content = message.get("content")
            if not content:
                continue
            if message["role"] == "user":
                widgets.append(Prompt(content))
            elif message["role"] == "assistant":
                reasoning, answer = split_thinking(self._for_display(content))
                if reasoning:
                    widgets.append(Reasoning(reasoning))
                answers.append(len(widgets))
                widgets.append(Response(answer))

        # Downgrade all but the newest answers to plain text. Done after the
        # loop so "newest" is known; a Response is swapped for a PlainAnswer
        # holding the same text, which is one widget instead of a hundred.
        for index in answers[:-REPLAY_RICH_TURNS]:
            widgets[index] = PlainAnswer(widgets[index].text)
        if widgets:
            await view.mount_all(widgets)
        view.scroll_end(animate=False)

    # ---- loading ----------------------------------------------------------

    @work(thread=True)
    def _load_components(self) -> None:
        try:
            engine = _build_engine(self._config_path, self._model_path,
                                   self._engine_name)
        except Exception as exc:
            # The spinner has to stop on the FAILURE path too, or a bad config
            # leaves the transcript spinning forever behind the error message.
            self.app.call_from_thread(self._stop_loading)
            self.app.call_from_thread(self._note,
                                      f"Could not start chat: {exc}", "error")
            self.app.call_from_thread(self._set_placeholder,
                                      "load failed; Esc to go back")
            return
        self._engine = engine
        self.app.call_from_thread(self._stop_loading)
        save_prefs(self._cwd, self._config_path, self._model_path)
        self.app.call_from_thread(
            self._note, f"Ready · {engine.describe()}", "ok")
        self.app.call_from_thread(self._mark_idle)

    # ---- leaving / cancelling ---------------------------------------------

    def action_back(self) -> None:
        # Esc closes the menu first, so opening it is not a one-way trip.
        palette = self.query_one("#chat-palette", OptionList)
        if palette.display:
            palette.display = False
            self.query_one("#chat-input", ChatInput).focus()
            return
        if self._busy:
            # First Esc stops the generation (the streamer sees the flag and
            # returns True to openvino_genai); the screen stays.
            self._stop_stream = True
            self._note("stopping generation…", "warn")
        else:
            self.app.pop_screen()

    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Show and copy a link rather than opening it.

        The href comes from the model, so following it on one click is not
        ours to decide. Copying puts the user one paste from their browser
        and lets them read where it goes first.
        """
        event.stop()
        self.app.copy_to_clipboard(event.href)
        self._note(f"link copied: {event.href}", "ok")

    # ---- the slash menu ----------------------------------------------------

    def on_text_area_changed(self, event) -> None:
        palette = self.query_one("#chat-palette", OptionList)
        value = event.text_area.text
        if value.startswith("/") and " " not in value:
            matches = [(name, help_text) for name, help_text in CHAT_COMMANDS
                       if name.startswith(value.lower())]
            palette.clear_options()
            if matches:
                palette.add_options([
                    Option(ui.slash_label(name, help_text, width=10),
                           id=name.lstrip("/"))
                    for name, help_text in matches
                ])
                palette.display = True
                return
        palette.display = False

    def on_key(self, event) -> None:
        """Down steps into the menu; otherwise Up/Down walk the history."""
        if event.key not in ("up", "down"):
            return
        palette = self.query_one("#chat-palette", OptionList)
        inp = self.query_one("#chat-input", ChatInput)
        if self.focused is not inp:
            return                       # the menu owns its own arrows
        if event.key == "down" and palette.display:
            palette.focus()
            if palette.option_count:
                palette.highlighted = 0
            event.stop()
            return
        if palette.display:
            return                       # browsing the menu, not the history
        if "\n" in inp.value:
            return                       # multi-line draft: arrows move the
            #                              cursor, as they must to edit it
        recalled = (self._history.previous(inp.value) if event.key == "up"
                    else self._history.next())
        if recalled is None:
            return
        inp.value = recalled
        self.call_after_refresh(setattr, inp, "cursor_position", len(recalled))
        event.stop()

    def on_option_list_option_selected(self,
                                       event: OptionList.OptionSelected) -> None:
        """Fill the line in, ready for an argument, rather than running it.

        /tokens, /save, /load and /copy all take one, so executing on
        selection would mean picking from the menu could never reach the
        useful form of most of these commands.
        """
        palette = self.query_one("#chat-palette", OptionList)
        inp = self.query_one("#chat-input", ChatInput)
        palette.display = False
        inp.value = f"/{event.option.id} "
        inp.focus()
        self.call_after_refresh(setattr, inp, "cursor_position", len(inp.value))

    # ---- the conversation --------------------------------------------------

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        line = event.value.strip()
        inp = self.query_one("#chat-input", ChatInput)
        inp.value = ""
        self.query_one("#chat-palette", OptionList).display = False
        if not line:
            return
        self._history.add(line)

        if line.startswith("/"):
            await self._slash(line)
            return

        if self._engine is None:
            self._note("Still loading (or load failed). One moment…", "warn")
            return
        if self._busy:
            self._note("Still answering; Esc stops it.", "warn")
            return

        self._busy = True
        self._set_placeholder("thinking…  Esc stops the answer")
        view = self._view
        await view.mount(Prompt(line))
        response = Response()
        await view.mount(response)
        # Anchoring pins this widget to the bottom of the scroll view, so a
        # long answer keeps itself in sight as it grows. This is what replaced
        # the separate streaming line and its tail-clipping.
        response.anchor()
        await view.mount(Thinking("● thinking…"))
        # Built HERE, on the main thread: get_stream starts a background task.
        self._ask(line, response, Markdown.get_stream(response))

    async def _slash(self, line: str) -> None:
        head, _, rest = line.partition(" ")
        name = rest.strip() or "last"
        if head in ("/back", "/exit"):
            if not self._busy:
                self.app.pop_screen()
            return
        if head == "/save":
            path = _session_path(self._cwd, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._session.save(path)
            self._note(f"saved conversation to {path}", "ok")
            return
        if head == "/load":
            # A bare /load opens the picker; /load <name> still goes direct,
            # so the shortcut survives for anyone who knows the name.
            if not rest.strip():
                self._pick_session()
                return
            await self._load_named(name)
            return
        if head == "/tokens":
            self._set_token_cap(rest.strip())
            return
        if head == "/copy":
            self._copy(rest.strip().lower())
            return
        if head in ("/thinking", "/think"):
            await self._toggle_thinking(rest.strip().lower())
            return
        if head == "/engine":
            self._switch_engine(rest.strip().lower())
            return
        known = " ".join(name for name, _ in CHAT_COMMANDS)
        self._note(f"unknown chat command {head}. I know {known}.", "warn")

    def _pick_session(self) -> None:
        """Open the picker, then load whatever it comes back with."""
        sessions = list_sessions(self._cwd)
        if not sessions:
            self._note("no saved conversations yet; /save one first.", "warn")
            return

        def chosen(name: str) -> None:
            if name:                       # "" means Esc, so do nothing
                self.run_worker(self._load_named(name))

        self.app.push_screen(SessionPicker(sessions), chosen)

    async def _load_named(self, name: str) -> None:
        if self._busy:
            self._note("still answering; Esc stops it before loading.", "warn")
            return
        path = _session_path(self._cwd, name)
        if not os.path.exists(path):
            self._note(f"no saved conversation at {path}", "warn")
            return
        try:
            self._session = Session.load(path)
        except (OSError, ValueError) as exc:
            # A hand-edited or truncated file is a bad load, not a crash.
            self._note(f"could not read {path}: {exc}", "error")
            return
        # Continue writing to the conversation you just opened, so adding to
        # an old chat extends it instead of forking a new autosave.
        self._autosave = name
        await self._replay()
        self._note(f"loaded {name} ({len(self._session.messages)} messages)",
                   "ok")

    def _set_token_cap(self, value: str) -> None:
        """/tokens N: retune the answer-length cap, 0 for none.

        The screen loaded the model with a 256-token cap and offered no way to
        change it, so a long answer was simply cut off mid-sentence. The
        provider re-reads max_new_tokens on every call, so setting it on the
        live object takes effect on the next question with NO model reload,
        which would otherwise cost ~30 seconds.
        """
        if self._engine is None:
            self._note("still loading; try /tokens once the model is ready.",
                       "warn")
            return
        try:
            cap = int(value)
        except ValueError:
            self._note(f"/tokens wants a number (0 = no cap), not {value!r}.",
                       "warn")
            return
        if cap < 0:
            self._note("/tokens cannot be negative. Use 0 for no cap.",
                       "warn")
            return
        self._engine.max_new_tokens = cap or None
        self._note(f"answer cap: {cap or 'no'} tokens", "ok")

    async def _toggle_thinking(self, value: str) -> None:
        """/thinking [on|off]: show or hide the model's reasoning blocks.

        Bare /thinking flips it. The transcript is rebuilt either way, so the
        reasoning you just hid actually leaves the screen.
        """
        if value in ("", "toggle"):
            self._show_thinking = not self._show_thinking
        elif value in ("on", "show", "yes"):
            self._show_thinking = True
        elif value in ("off", "hide", "no"):
            self._show_thinking = False
        else:
            self._note(f"/thinking takes on, off, or nothing, not {value!r}.",
                       "warn")
            return
        if self._busy:
            # The redraw would remove the answer being streamed into.
            self._note("still answering; the change applies once it lands.",
                       "warn")
            return
        await self._replay()
        state = "shown" if self._show_thinking else "hidden"
        self._note(f"reasoning is now {state}.", "ok")

    def _switch_engine(self, value: str) -> None:
        """/engine [local|ovms]: pick which backend answers.

        local  openvino_genai in this process. Retrieval then one answer, and
               no tool calling, because a bare pipeline has no way to be told
               about tools.
        ovms   the same agent `ovat run` builds, against the server named in
               the workflow's model: section. This is where ovms_url and
               tool_parser finally do something, and where search_docs,
               transcribe and any mcp_stdio tools become reachable from chat.

        Switching rebuilds, which for local means paying the model load again,
        so it is a deliberate command rather than something inferred.
        """
        if value in ("", "?"):
            current = (self._engine.describe() if self._engine
                       else f"{self._engine_name} (still loading)")
            self._note(f"engine: {current}")
            self._note("switch with /engine local  or  /engine ovms")
            return
        if value not in ("local", "ovms"):
            self._note(f"/engine takes local or ovms, not {value!r}.", "warn")
            return
        if value == self._engine_name and self._engine is not None:
            self._note(f"already on {value}.")
            return
        if self._busy:
            self._note("still answering; Esc stops it first.", "warn")
            return

        if self._engine is not None:
            try:
                self._engine.close()
            except Exception as exc:
                # A retriever that will not close (a locked sqlite file, say)
                # must not strand the user on an engine they asked to leave.
                self._note(f"note: closing the old engine failed ({exc})",
                           "warn")
        self._engine = None
        self._engine_name = value
        self._note(f"switching to {value}…")
        self._view.loading = True
        self._set_placeholder("loading the engine…")
        self._load_components()

    def _last(self, role: str) -> str:
        """The most recent message from `role`, or "" if there is none yet."""
        for message in reversed(self._session.messages):
            if message.get("role") == role and message.get("content"):
                return message["content"]
        return ""

    def _copy(self, what: str) -> None:
        """/copy [me|all]: the last answer, your last question, or everything.

        A terminal's own selection is awkward for a long streamed answer that
        wrapped over dozens of lines, which is exactly the output worth
        keeping. The transcript already holds every turn, so copying from it
        beats dragging a mouse across a scrollback.
        """
        # Copying follows what is ON SCREEN, so with reasoning hidden you get
        # the answer, and /thinking on then /copy gets the reasoning too.
        if what in ("", "last", "answer", "it"):
            text = self._for_display(self._last("assistant"))
            label = "the last answer"
        elif what in ("me", "mine", "my", "user", "question"):
            text, label = self._last("user"), "your last question"
        elif what in ("all", "everything", "chat"):
            text = "\n\n".join(
                f"{m['role']}: "
                f"{self._for_display(m['content']) if m['role'] == 'assistant' else m['content']}"
                for m in self._session.messages if m.get("content"))
            label = "the whole conversation"
        else:
            self._note(f"/copy takes nothing, 'me' or 'all', not {what!r}.",
                       "warn")
            return

        if not text:
            self._note("nothing to copy yet.", "warn")
            return
        self.app.copy_to_clipboard(text)
        self._note(f"copied {label} ({len(text)} chars).", "ok")

    def _clear_thinking(self) -> None:
        """Remove the pulse. Safe to call twice; the error path also calls it."""
        for widget in list(self.query(Thinking)):
            widget.remove()

    def _commit_turn(self, response: "Response", answer: str,
                     sources: list) -> None:
        """Put the finished answer in place, in ONE main-thread pass."""
        self._clear_thinking()
        reasoning, shown = split_thinking(self._for_display(answer))
        if answer and not shown:
            # Everything that came back was reasoning, so the cap ran out
            # before the model reached its answer. Say which knobs help.
            shown = ("*only reasoning came back before generation stopped; "
                     "/thinking on to read it, or raise /tokens*")
        # The answer widget can be GONE by the time the answer arrives:
        # /thinking and /load both rebuild the transcript, and if either ran
        # mid-generation this Response was removed with everything else.
        # Mounting relative to an orphan raises MountError and takes the app
        # down, so check for a parent and fall back to appending.
        alive = response.parent is not None
        if reasoning:
            if alive:
                self._view.mount(Reasoning(reasoning), before=response)
            else:
                self._view.mount(Reasoning(reasoning))
        if alive:
            response.update(shown)
        else:
            self._view.mount(Response(shown))
        if sources:
            label = self._engine.footnote_label if self._engine else "sources"
            footnote = Sources(f"{label}: " + ", ".join(sources))
            if alive:
                self._view.mount(footnote, after=response)
            else:
                self._view.mount(footnote)
        self._mark_idle()

    @work(thread=True)
    def _ask(self, question: str, response: "Response", stream) -> None:
        engine = self._engine
        parts: list = []
        written = ""                     # how much of the answer is on screen

        def on_token(token: str):
            nonlocal written
            if self._stop_stream:
                return True              # tell openvino_genai to stop generating
            parts.append(token)
            _, answer = split_thinking(self._for_display("".join(parts)))
            if answer == written:
                return False             # still inside <think>: nothing new yet
            if answer.startswith(written):
                # The common case: append only what arrived. MarkdownStream
                # coalesces bursts internally, which is what its docs say the
                # plain update() path cannot keep up with past ~20 a second.
                self.app.call_from_thread(stream.write, answer[len(written):])
            else:
                # Stripping reshaped the text (an unclosed <think> just
                # closed), so append-only no longer holds: replace it.
                self.app.call_from_thread(response.update, answer)
            written = answer
            return False

        try:
            answer, sources = engine.ask(question, self._session.messages,
                                         on_token)
        except Exception as exc:
            self.app.call_from_thread(stream.stop)
            self.app.call_from_thread(self._clear_thinking)
            self.app.call_from_thread(response.update, f"*error: {exc}*")
            self.app.call_from_thread(self._mark_idle)
            return

        answer = (answer or "").strip()
        self._session.add_user(question)
        self._session.add_assistant(answer)
        # Autosave after every turn, so a crash never loses the conversation.
        path = _session_path(self._cwd, self._autosave)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._session.save(path)

        # Stop the stream BEFORE the final update, so the background task is
        # not still appending while the finished text is written in.
        self.app.call_from_thread(stream.stop)
        self.app.call_from_thread(self._commit_turn, response, answer, sources)

