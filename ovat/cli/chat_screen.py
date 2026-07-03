# ovat/cli/chat_screen.py
"""The native chat screen: talk to your indexed documents inside the TUI.

Unlike every other TUI command (which runs `ovat ...` as a subprocess), chat
runs IN-PROCESS. That is what makes it feel like a chat app: the model loads
once and stays warm, replies stream token by token, and the conversation has
real memory (prior turns go back into the prompt via rag_chat's history).

Layout: a transcript log, a live "streaming" line that fills as the model
generates, and an input. Esc stops a generation in flight, or leaves the
screen when idle. /save and /load persist the conversation as JSON under
.ovat/sessions/ — finally putting Session.save/load to work.

The heavy lifting (config → retriever + local LLM) sits behind the
_build_components seam so tests can swap in fakes and drive the whole screen
headless in milliseconds.
"""
import json
import os

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
            "this workflow has no rag: section — add one and run `ovat index` first"
        )
    # Identify BEFORE the 30-second load. A vision/whisper/embedding folder
    # used to load "fine" and then explode at generate time with a C++
    # traceback about tensor ports — now it is one readable sentence.
    kind, why = identify_model(model_path)
    if kind not in ("llm", "unknown"):
        _, llms = pick_chat_llm()
        suggestion = (" Try: " + ", ".join(m["name"] for m in llms)
                      if llms else "")
        raise ValueError(
            f"{os.path.basename(model_path.rstrip(os.sep))} is not a text "
            f"LLM ({why}) — chat needs a text model.{suggestion}"
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
    #chat-stream { height: auto; margin: 0 2; padding: 0 1; }
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
        self._set_placeholder("ask about your docs  ·  /save /load /back  ·  "
                              "Esc stops / leaves")

    # ---- loading ----------------------------------------------------------

    @work(thread=True)
    def _load_components(self) -> None:
        try:
            components = _build_components(self._config_path, self._model_path)
        except Exception as exc:
            self.app.call_from_thread(self._log, Text(f"Could not start chat: {exc}",
                                                  style=ui.RED))
            self.app.call_from_thread(self._set_placeholder, "load failed — Esc to go back")
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
            self._log(Text("Still answering — Esc stops it.", style=ui.YELLOW))
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
            self._log(Text(f"— loaded {path} —", style=ui.DIM))
            for message in self._session.messages:
                if message["role"] == "user":
                    self._log(Text(f"you › {message['content']}",
                                   style=f"bold {ui.CYAN}"))
                elif message["role"] == "assistant" and message.get("content"):
                    self._log(Text(f"ovat › {message['content']}"))
            return
        self._log(Text(f"unknown chat command {head}. I know /save /load /back.",
                       style=ui.YELLOW))

    @work(thread=True)
    def _ask(self, question: str) -> None:
        cfg, retriever, llm = self._components
        stream_line = self.query_one("#chat-stream", Static)
        parts: list = []

        def on_token(token: str):
            if self._stop_stream:
                return True              # tell openvino_genai to stop generating
            parts.append(token)
            self.app.call_from_thread(stream_line.update,
                                  Text("ovat › " + "".join(parts), style=ui.DIM))
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

        self.app.call_from_thread(stream_line.update, "")
        self.app.call_from_thread(self._log, Text(f"ovat › {answer}"))
        if sources:
            self.app.call_from_thread(self._log, Text("sources: " + ", ".join(sources),
                                                  style=ui.DIM))
        self.app.call_from_thread(self._mark_idle)
