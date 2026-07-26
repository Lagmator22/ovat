# ovat/cli/chat_screen.py
"""The native chat screen: talk to your indexed documents inside the TUI.

Unlike every other TUI command (which runs `ovat ...` as a subprocess), chat
runs IN-PROCESS. That is what makes it feel like a chat app: the model loads
once and stays warm, replies stream token by token, and the conversation has
real memory (prior turns go back into the prompt via rag_chat's history).

Layout: a transcript log, a live "streaming" line that fills as the model
generates, and an input. Esc stops a generation in flight, or leaves the
screen when idle. /save and /load persist the conversation as JSON under
.ovat/sessions/, finally putting Session.save/load to work.

The heavy lifting (config → retriever + local LLM) sits behind the
_build_components seam so tests can swap in fakes and drive the whole screen
headless in milliseconds.
"""
import json
import os
import time

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Input, RichLog, Static

from ovat.agent.rag_chat import rag_chat
from ovat.agent.session import Session
from ovat.cli import ui

# Both live under .ovat/ in the directory the user launched from, so a
# project's chats and preferences stay with that project.
PREFS_FILE = "chat_prefs.json"
SESSIONS_DIR = "sessions"

# The live "streaming" line is a preview of the answer in flight, NOT the
# transcript; the finished answer goes to the log. So it renders only the TAIL
# of what has arrived so far. Without this the widget grew one row per wrapped
# line: a 400-token answer collapsed the transcript to zero rows and pushed the
# input off the bottom of the screen. #chat-stream's max-height is the hard
# guarantee; this budget keeps the content near that size.
STREAM_TAIL_CHARS = 320
# Repainting on every token means one blocking main-thread hop per token, which
# throttles generation to the UI's refresh rate. Sample instead, the same trick
# shell.iter_display_lines uses for \r progress frames. Dropped frames cost
# nothing: the full answer is committed to the log when generation ends.
STREAM_REFRESH_S = 0.05


def _stream_tail(parts: list) -> Text:
    """The live line: the last STREAM_TAIL_CHARS of the answer so far."""
    text = "".join(parts)
    clipped = text[-STREAM_TAIL_CHARS:]
    prefix = "ovat › " if len(clipped) == len(text) else "ovat › …"
    return Text(prefix + clipped, style=ui.DIM)


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


def _build_components(config_path: str, model_path: str, max_tokens: int = 256):
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


class ChatScreen(Screen):
    """One conversation with the indexed documents, streaming and stateful."""

    DEFAULT_CSS = """
    ChatScreen { background: #0b0e14; }
    #chat-header { padding: 0 2; height: auto; }
    #chat-log {
        height: 1fr;
        border: round #0068B5;
        background: #0d1117;
        padding: 0 1;
        margin: 1 2 0 2;
    }
    /* max-height is load-bearing: the live line must never grow enough to
       squeeze the 1fr transcript or push the input off the screen. */
    #chat-stream { height: auto; max-height: 6; margin: 0 2; padding: 0 1; }
    #chat-input { margin: 0 2 1 2; border: round #8F5CFF; }
    """

    BINDINGS = [Binding("escape", "back", show=False)]

    def __init__(self, config_path: str, model_path: str, cwd: str | None = None):
        super().__init__()
        self._config_path = config_path
        self._model_path = model_path
        self._cwd = cwd or os.getcwd()
        self._components = None          # (cfg, retriever, llm) once loaded
        self._session = Session()        # the transcript; system stays in rag_chat
        self._busy = False
        self._stop_stream = False        # Esc mid-generation sets this

    def compose(self) -> ComposeResult:
        header = Text()
        header.append("OVAT chat", style=f"bold {ui.CYAN}")
        header.append(f"  ·  {os.path.basename(self._config_path)}"
                      f"  ·  {os.path.basename(self._model_path)}", style=ui.DIM)
        yield Static(header, id="chat-header")
        yield RichLog(id="chat-log", highlight=False, markup=False, wrap=True)
        yield Static("", id="chat-stream")
        yield Input(placeholder="loading the model…", id="chat-input")

    def on_mount(self) -> None:
        self.query_one("#chat-input", Input).focus()
        self._log(Text("Loading retriever + local model (first load can take a "
                       "while)…", style=ui.DIM))
        self._load_components()

    # ---- helpers ----------------------------------------------------------

    def _log(self, text: Text) -> None:
        self.query_one("#chat-log", RichLog).write(text)

    def _set_placeholder(self, text: str) -> None:
        self.query_one("#chat-input", Input).placeholder = text

    def _mark_idle(self) -> None:
        self._busy = False
        self._stop_stream = False
        self._set_placeholder("ask about your docs  ·  /save /load /tokens "
                              "/copy /back  ·  Esc stops / leaves")

    def _commit_turn(self, answer: str, sources: list) -> None:
        """Retire the live line and commit the answer, in ONE main-thread pass.

        Clearing the stream and writing the log used to be separate
        call_from_thread hops, so the screen repainted with neither showing:
        a visible flicker between the streamed text vanishing and the answer
        appearing. Batched, the swap happens within a single refresh.
        """
        self.query_one("#chat-stream", Static).update("")
        self._log(Text(f"ovat › {answer}"))
        if sources:
            self._log(Text("sources: " + ", ".join(sources), style=ui.DIM))
        self._mark_idle()

    # ---- loading ----------------------------------------------------------

    @work(thread=True)
    def _load_components(self) -> None:
        try:
            components = _build_components(self._config_path, self._model_path)
        except Exception as exc:
            self.app.call_from_thread(self._log, Text(f"Could not start chat: {exc}",
                                                  style=ui.RED))
            self.app.call_from_thread(self._set_placeholder, "load failed; Esc to go back")
            return
        self._components = components
        save_prefs(self._cwd, self._config_path, self._model_path)
        self.app.call_from_thread(self._log, Text("Ready. Ask a question about your "
                                              "indexed documents.", style=ui.GREEN))
        self.app.call_from_thread(self._mark_idle)

    # ---- leaving / cancelling ---------------------------------------------

    def action_back(self) -> None:
        if self._busy:
            # First Esc stops the generation (the streamer sees the flag and
            # returns True to openvino_genai); the screen stays.
            self._stop_stream = True
            self._log(Text("stopping generation…", style=ui.YELLOW))
        else:
            self.app.pop_screen()

    # ---- the conversation --------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        inp = self.query_one("#chat-input", Input)
        inp.value = ""
        if not line:
            return

        if line.startswith("/"):
            self._slash(line)
            return

        if self._components is None:
            self._log(Text("Still loading (or load failed). One moment…",
                           style=ui.YELLOW))
            return
        if self._busy:
            self._log(Text("Still answering; Esc stops it.", style=ui.YELLOW))
            return

        self._busy = True
        self._set_placeholder("thinking…  Esc stops the answer")
        self._log(Text(f"you › {line}", style=f"bold {ui.CYAN}"))
        self._ask(line)

    def _slash(self, line: str) -> None:
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
            self._log(Text(f"saved conversation to {path}", style=ui.GREEN))
            return
        if head == "/load":
            path = _session_path(self._cwd, name)
            if not os.path.exists(path):
                self._log(Text(f"no saved conversation at {path}", style=ui.YELLOW))
                return
            self._session = Session.load(path)
            self._log(Text(f"(loaded {path})", style=ui.DIM))
            for message in self._session.messages:
                if message["role"] == "user":
                    self._log(Text(f"you › {message['content']}",
                                   style=f"bold {ui.CYAN}"))
                elif message["role"] == "assistant" and message.get("content"):
                    self._log(Text(f"ovat › {message['content']}"))
            return
        if head == "/tokens":
            self._set_token_cap(rest.strip())
            return
        if head == "/copy":
            self._copy(rest.strip().lower())
            return
        self._log(Text(f"unknown chat command {head}. "
                       f"I know /save /load /tokens /copy /back.",
                       style=ui.YELLOW))

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
        if what in ("", "last", "answer", "it"):
            text, label = self._last("assistant"), "the last answer"
        elif what in ("me", "mine", "my", "user", "question"):
            text, label = self._last("user"), "your last question"
        elif what in ("all", "everything", "chat"):
            text = "\n\n".join(
                f"{m['role']}: {m['content']}"
                for m in self._session.messages if m.get("content"))
            label = "the whole conversation"
        else:
            self._log(Text(f"/copy takes nothing, 'me' or 'all', not {what!r}.",
                           style=ui.YELLOW))
            return

        if not text:
            self._log(Text("nothing to copy yet.", style=ui.YELLOW))
            return
        self.app.copy_to_clipboard(text)
        self._log(Text(f"copied {label} ({len(text)} chars).", style=ui.GREEN))

    def _set_token_cap(self, value: str) -> None:
        """/tokens N: retune the answer-length cap, 0 for none.

        The screen loaded the model with a 256-token cap and offered no way to
        change it, so a long answer was simply cut off mid-sentence. The
        provider re-reads max_new_tokens on every call, so setting it on the
        live object takes effect on the next question with NO model reload,
        which would otherwise cost ~30 seconds.
        """
        if self._components is None:
            self._log(Text("still loading; try /tokens once the model is ready.",
                           style=ui.YELLOW))
            return
        try:
            cap = int(value)
        except ValueError:
            self._log(Text(f"/tokens wants a number (0 = no cap), not {value!r}.",
                           style=ui.YELLOW))
            return
        if cap < 0:
            self._log(Text("/tokens cannot be negative. Use 0 for no cap.",
                           style=ui.YELLOW))
            return
        self._components[2].max_new_tokens = cap or None
        self._log(Text(f"answer cap: {cap or 'no'} tokens", style=ui.GREEN))

    @work(thread=True)
    def _ask(self, question: str) -> None:
        cfg, retriever, llm = self._components
        stream_line = self.query_one("#chat-stream", Static)
        parts: list = []
        last_render = 0.0

        def on_token(token: str):
            nonlocal last_render
            if self._stop_stream:
                return True              # tell openvino_genai to stop generating
            parts.append(token)
            now = time.monotonic()
            if now - last_render < STREAM_REFRESH_S:
                return False             # too soon to repaint; the token is kept
            last_render = now
            self.app.call_from_thread(stream_line.update, _stream_tail(parts))
            return False

        try:
            answer, sources = rag_chat(
                retriever, llm, question,
                top_k=4,
                system_prompt=cfg.agent.system_prompt,
                history=self._session.messages,      # turns BEFORE this question
                on_token=on_token,
            )
        except Exception as exc:
            self.app.call_from_thread(stream_line.update, "")
            self.app.call_from_thread(self._log, Text(f"error: {exc}", style=ui.RED))
            self.app.call_from_thread(self._mark_idle)
            return

        answer = (answer or "").strip()
        self._session.add_user(question)
        self._session.add_assistant(answer)
        # Autosave after every turn, so a crash never loses the conversation.
        path = _session_path(self._cwd, "last")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._session.save(path)

        self.app.call_from_thread(self._commit_turn, answer, sources)
