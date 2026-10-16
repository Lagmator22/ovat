# ovat/tui/server/bridge.py
"""OVAT's agent events, translated into the history events the copied TUI
renders (ovat/tui/vibe/app_server/events.py and models.py).

The copied event handler draws tool calls, results and answers from
PublicEffectEntry / PublicMessageEntry objects. OVAT's agent reports
tool_started / tool_ended / subagent_status / remote_status and a final
answer. This is the one place that maps one onto the other, so the copied
widgets stay unchanged.

The app server (ovat/tui/server/projector.py) uses it to fill the session it
publishes: it sets `session_id` and `turn_id` so every entry names the
conversation and the turn it belongs to.
"""
import itertools
import time

from ovat.agent.events import (REMOTE_STATUS, SUBAGENT_STATUS, TOOL_ENDED,
                               TOOL_STARTED, Event)
from ovat.tui.vibe.app_server._effect_models import (
    FileEditEffectDetail, FileEditEffectInput, FileEditEffectOutput,
    FileReadEffectDetail, FileReadEffectInput, FileReadEffectOutput,
    FileSearchEffectDetail, FileSearchEffectInput, FileSearchEffectMatch,
    FileSearchEffectOutput, FileWriteEffectDetail, FileWriteEffectInput,
    FileWriteEffectOutput, GenericEffectDetail, ShellEffectDetail,
    ShellEffectInput, ShellEffectOutput, SubagentEffectDetail,
    SubagentEffectInput)
from ovat.tui.vibe.app_server.events import (HistoryEntryAdded,
                                             HistoryEntryUpdated)
from ovat.tui.vibe.app_server.models import (CompletedEffectState,
                                             FailedEffectState,
                                             JsonPatchOperation,
                                             PublicEffectEntry,
                                             PublicEntryGenerationStatus,
                                             PublicError, PublicMessageEntry,
                                             PublicReasoningEntry,
                                             RunningEffectState,
                                             SkippedEffectState,
                                             TextContentBlock)
from ovat.tui.vibe.utils.tool_presentation import (EffectCallDisplay,
                                                   EffectResultDisplay)

SESSION_ID = "ovat"

# (running verb, settled verb, status line) per built-in tool.
_WORDS = {
    "search_docs": ("Searching", "Searched", "Searching docs"),
    "transcribe": ("Transcribing", "Transcribed", "Transcribing audio"),
    "describe_image": ("Looking at", "Looked at", "Looking at an image"),
    # The coding tools; the words are Vibe's own (read_file.py, write_file.py,
    # edit.py, grep.py, bash.py).
    "read_file": ("Reading", "Read", "Reading file"),
    "write_file": ("Creating", "Created", "Writing file"),
    "edit": ("Editing", "Edited", "Editing files"),
    "grep": ("Searching", "Searched", "Searching files"),
    "shell": ("Running", "Ran", "Running command"),
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _first_arg(arguments: dict) -> str:
    if not arguments:
        return ""
    value = next(iter(arguments.values()))
    text = " ".join(str(value).split())
    return text if len(text) <= 80 else text[:79] + "…"


def _coding_detail(name: str, a: dict, display):
    """The Vibe effect card for a coding tool call, from its arguments."""
    try:
        if name == "read_file":
            return FileReadEffectDetail(
                tool_name=name, display=display, input=FileReadEffectInput(
                    file_path=str(a.get("file_path", "")),
                    offset=a.get("offset"),
                    limit=a.get("limit") or FileReadEffectInput.DEFAULT_LIMIT))
        if name == "write_file":
            return FileWriteEffectDetail(
                tool_name=name, display=display, input=FileWriteEffectInput(
                    file_path=str(a.get("file_path", "")),
                    content=str(a.get("content", ""))))
        if name == "edit":
            return FileEditEffectDetail(
                tool_name=name, display=display, input=FileEditEffectInput(
                    file_path=str(a.get("file_path", "")),
                    old_string=str(a.get("old_string", "")),
                    new_string=str(a.get("new_string", "")),
                    replace_all=bool(a.get("replace_all", False))))
        if name == "grep":
            return FileSearchEffectDetail(
                tool_name=name, display=display, input=FileSearchEffectInput(
                    pattern=str(a.get("pattern", "")),
                    path=str(a.get("path") or "."),
                    max_matches=a.get("max_matches")))
        if name == "shell":
            return ShellEffectDetail(
                tool_name=name, display=display,
                input=ShellEffectInput(command=str(a.get("command", ""))))
    except (ValueError, TypeError):
        # A model that sends a number where a string belongs still gets a
        # card, just the generic one.
        return None
    return None


def _coding_output(shown: dict):
    """The Vibe output model for what a coding tool reported, or None."""
    kind = shown.get("kind")
    if kind == "shell":
        stdout, stderr = shown.get("stdout", ""), shown.get("stderr", "")
        output = ShellEffectOutput(
            stdout=stdout, stderr=stderr, output=stdout + (
                ("\n" if stdout and not stdout.endswith("\n") and stderr
                 else "") + stderr),
            truncated=bool(shown.get("truncated")))
    elif kind == "edit":
        output = FileEditEffectOutput(
            file=shown["file"], old_string=shown.get("old_string"),
            new_string=shown.get("new_string"), occurrences=shown.get(
                "occurrences") or [])
    elif kind == "write_file":
        output = FileWriteEffectOutput(
            file_path=shown["file_path"], content=shown.get("content", ""),
            file_existed=bool(shown.get("file_existed")),
            previous_content=shown.get("previous_content"))
    elif kind == "read_file":
        output = FileReadEffectOutput(
            file_path=shown["file_path"], content=shown.get("content", ""),
            num_lines=shown.get("num_lines", 0),
            start_line=shown.get("start_line", 1),
            requested_offset=shown.get("requested_offset"),
            requested_limit=shown.get("requested_limit"),
            total_lines=shown.get("total_lines"),
            was_truncated=bool(shown.get("was_truncated")))
    elif kind == "grep":
        output = FileSearchEffectOutput(
            matches=shown.get("matches", ""),
            match_count=shown.get("match_count", 0),
            was_truncated=bool(shown.get("was_truncated")),
            parsed_matches=[FileSearchEffectMatch(**m) for m in
                            shown.get("parsed_matches") or []])
    else:
        return None
    return output.model_dump(mode="json", by_alias=True)


def _coding_text(shown: dict) -> str:
    """A short plain line for the settled card: what the tool did."""
    kind = shown.get("kind")
    if kind == "shell":
        return shown.get("command", "")
    if kind == "grep":
        n = shown.get("match_count", 0)
        return f"{shown.get('pattern', '')} ({n} match{'' if n == 1 else 'es'})"
    if kind == "read_file":
        n = shown.get("num_lines", 0)
        return f"{n} line{'' if n == 1 else 's'} from {shown['file_path']}"
    return shown.get("file") or shown.get("file_path") or ""


class Bridge:
    def __init__(self, session_id: str = SESSION_ID, first_id: int = 1) -> None:
        self.session_id = session_id
        # The turn the next entries belong to. None outside a turn (and for
        # the classic agent TUI, which has no turns).
        self.turn_id: str | None = None
        self._entries: dict[str, PublicEffectEntry] = {}
        self._ids = itertools.count(first_id)
        self._last_running: str | None = None

    def running(self, tool: str) -> PublicEffectEntry | None:
        """The newest tool call still running under this name, if any."""
        for entry in reversed(list(self._entries.values())):
            if entry.title == tool:
                return entry
        return None

    # ---- tools ---------------------------------------------------------------

    def _detail(self, name: str, kind: str, arguments: dict):
        if kind in ("subagent", "a2a"):
            agent = arguments.get("agent") or name
            task = arguments.get("task") or arguments.get("message") or ""
            display = EffectCallDisplay(
                summary=f"{agent}: {task}", verb="Delegating to",
                message=agent, settled_verb="Delegated to",
                settled_message=agent, status_text=f"Delegating to {agent}")
            return SubagentEffectDetail(
                tool_name=name, display=display, agent_name=agent,
                input=SubagentEffectInput(task=task, agent=agent))
        verb, settled, status = _WORDS.get(
            name, ("Calling", "Called", f"Calling {name}"))
        message = _first_arg(arguments) or name
        display = EffectCallDisplay(
            summary=f"{name}({message})", verb=verb, message=message,
            settled_verb=settled, settled_message=message, status_text=status)
        if name == "search_docs":
            return FileSearchEffectDetail(tool_name=name, display=display)
        if name in ("transcribe", "describe_image"):
            return FileReadEffectDetail(tool_name=name, display=display)
        coding = _coding_detail(name, arguments, display)
        if coding is not None:
            return coding
        return GenericEffectDetail(tool_name=name, display=display,
                                   input=arguments or None)

    def tool_started(self, event: Event, kind: str) -> HistoryEntryAdded:
        d = event.data
        call_id = d.get("call_id") or f"call-{next(self._ids)}"
        entry = PublicEffectEntry(
            id=call_id, session_id=self.session_id, turn_id=self.turn_id,
            created_at=_now_ms(), updated_at=_now_ms(),
            generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
            title=d.get("name", "tool"),
            detail=self._detail(d.get("name", "tool"), kind,
                                d.get("arguments") or {}),
            state=RunningEffectState())
        self._entries[call_id] = entry
        self._last_running = call_id
        return HistoryEntryAdded(entry=entry)

    def tool_ended(self, event: Event) -> HistoryEntryUpdated | None:
        d = event.data
        previous = self._entries.pop(d.get("call_id"), None)
        if previous is None:
            return None
        status = d.get("status", "ok")
        duration_ms = float(d.get("duration_s") or 0) * 1000
        display = previous.detail.display
        shown = d.get("display") or {}
        # A failed call has no output to show, except a command that ran and
        # exited badly: its stdout and stderr are the point.
        has_output = shown and (not shown.get("failed")
                                or shown.get("kind") == "shell")
        output = _coding_output(shown) if has_output else None
        if status == "ok":
            message = _coding_text(shown) or display.settled_message \
                or display.message or ""
            state = CompletedEffectState(
                duration_ms=duration_ms, output=output,
                display=EffectResultDisplay(
                    success=True, verb=display.settled_verb, message=message))
        elif status == "denied":
            state = SkippedEffectState(
                reason="not allowed",
                display=EffectResultDisplay(success=False, verb="Skipped",
                                            message="not allowed"))
        else:
            # A coding tool says why in its own sentence ("Error: ...").
            reason = shown.get("error") or f"{previous.title} failed"
            state = FailedEffectState(
                duration_ms=duration_ms, output=output,
                error=PublicError(message=reason),
                display=EffectResultDisplay(success=False, verb="Failed",
                                            message=reason if shown.get(
                                                "error") else previous.title))
        entry = previous.model_copy(update={
            "state": state, "updated_at": _now_ms(),
            "generation_status": PublicEntryGenerationStatus.COMPLETED})
        if self._last_running == entry.id:
            self._last_running = None
        return HistoryEntryUpdated(previous=previous, entry=entry, patch=[])

    def progress(self, event: Event) -> HistoryEntryUpdated | None:
        """A subagent's or remote agent's progress line, shown under the
        running call that started it."""
        call_id = self._last_running
        previous = self._entries.get(call_id) if call_id else None
        if previous is None:
            return None
        d = event.data
        text = f"{d.get('agent')}: {d.get('text')}"
        return HistoryEntryUpdated(
            previous=previous, entry=previous,
            patch=[JsonPatchOperation(op="append", path="/state/outputText",
                                      value=text)])

    def convert(self, event: Event, kinds: dict[str, str]):
        """The history event for one agent event, or None."""
        if event.type == TOOL_STARTED:
            return self.tool_started(event, kinds.get(
                event.data.get("name"), "builtin"))
        if event.type == TOOL_ENDED:
            return self.tool_ended(event)
        if event.type in (SUBAGENT_STATUS, REMOTE_STATUS):
            return self.progress(event)
        return None

    # ---- answers -------------------------------------------------------------

    def answer(self, text: str, status: PublicEntryGenerationStatus = (
               PublicEntryGenerationStatus.COMPLETED)) -> HistoryEntryAdded:
        entry = PublicMessageEntry(
            id=f"msg-{next(self._ids)}", session_id=self.session_id,
            turn_id=self.turn_id, created_at=_now_ms(),
            updated_at=_now_ms(), generation_status=status,
            role="assistant", content=[TextContentBlock(text=text)])
        return HistoryEntryAdded(entry=entry)

    def reasoning(self, text: str, status: PublicEntryGenerationStatus = (
                  PublicEntryGenerationStatus.COMPLETED)) -> HistoryEntryAdded:
        """What a thinking model reasoned before it answered."""
        entry = PublicReasoningEntry(
            id=f"reasoning-{next(self._ids)}", session_id=self.session_id,
            turn_id=self.turn_id, created_at=_now_ms(),
            updated_at=_now_ms(), generation_status=status,
            text=text)
        return HistoryEntryAdded(entry=entry)
