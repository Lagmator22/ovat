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
from ovat.tui.vibe.app_server._effect_models import (FileReadEffectDetail,
                                                     FileSearchEffectDetail,
                                                     GenericEffectDetail,
                                                     SubagentEffectDetail,
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
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _first_arg(arguments: dict) -> str:
    if not arguments:
        return ""
    value = next(iter(arguments.values()))
    text = " ".join(str(value).split())
    return text if len(text) <= 80 else text[:79] + "…"


class Bridge:
    def __init__(self, session_id: str = SESSION_ID) -> None:
        self.session_id = session_id
        # The turn the next entries belong to. None outside a turn (and for
        # the classic agent TUI, which has no turns).
        self.turn_id: str | None = None
        self._entries: dict[str, PublicEffectEntry] = {}
        self._ids = itertools.count(1)
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
        if status == "ok":
            state = CompletedEffectState(
                duration_ms=duration_ms,
                display=EffectResultDisplay(
                    success=True, verb=display.settled_verb,
                    message=display.settled_message or display.message or ""))
        elif status == "denied":
            state = SkippedEffectState(
                reason="not allowed",
                display=EffectResultDisplay(success=False, verb="Skipped",
                                            message="not allowed"))
        else:
            state = FailedEffectState(
                duration_ms=duration_ms,
                error=PublicError(message=f"{previous.title} failed"),
                display=EffectResultDisplay(success=False, verb="Failed",
                                            message=previous.title))
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

    def answer(self, text: str) -> HistoryEntryAdded:
        entry = PublicMessageEntry(
            id=f"msg-{next(self._ids)}", session_id=self.session_id,
            turn_id=self.turn_id, created_at=_now_ms(),
            updated_at=_now_ms(),
            generation_status=PublicEntryGenerationStatus.COMPLETED,
            role="assistant", content=[TextContentBlock(text=text)])
        return HistoryEntryAdded(entry=entry)

    def reasoning(self, text: str) -> HistoryEntryAdded:
        """What a thinking model reasoned before it answered."""
        entry = PublicReasoningEntry(
            id=f"reasoning-{next(self._ids)}", session_id=self.session_id,
            turn_id=self.turn_id, created_at=_now_ms(),
            updated_at=_now_ms(),
            generation_status=PublicEntryGenerationStatus.COMPLETED,
            text=text)
        return HistoryEntryAdded(entry=entry)
