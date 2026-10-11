# ovat/agent/events.py
"""Layer 3.5: what an agent is DOING, told to whoever wants to know.

Note to myself: `run(text) -> text` tells the caller what the agent ended
with, never what it did on the way. A UI that wants a live "calling
search_docs" card, and an A2A server that wants to stream progress to a
remote caller, both need the in-between. This file is that channel.

Two pieces:

- EventBus: one per built agent. Listeners subscribe; the agent's code
  emits. It is the observer pattern, and in C++ terms the bus is a vector of
  std::function callbacks plus a mutex. It also carries the cancel flag
  (`stop`) and a small `context` dict for facts about the run in progress
  (the A2A hop count), because every tool call already passes through here.

- observed(): wraps ONE tool function. All four engines call the same
  `tools[name]["function"]` (native loop, LangChain, LlamaIndex, Agents SDK),
  so wrapping it once in factory.build_tools() gives every engine tool
  events and a cancel check without touching any engine file.

Rules that keep this safe:
- stdlib only. The CLI, the TUI and the A2A server all import it; none of
  them may pull a framework or textual in through here (isolation, rule 4).
- A listener that raises never breaks the agent: events are best effort.
- Listeners run on the agent's thread. A UI must hand the event to its own
  thread (Textual: post_message) instead of drawing from here.
- With no listener the agent behaves exactly as it did before this existed.
"""
import functools
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

# Event types. Each one documents its `data` fields. A new type is an
# addition here; a listener ignores types it does not know, so adding one
# never breaks an existing listener.
#
#   tool_started  call_id, name, arguments
#   tool_ended    call_id, name, status ("ok" | "error" | "cancelled"),
#                 duration_s, result_chars
#   remote_status agent, state, text      (a remote A2A agent's progress)
TOOL_STARTED = "tool_started"
TOOL_ENDED = "tool_ended"
REMOTE_STATUS = "remote_status"

# What a cancelled tool hands the model instead of running. One sentence,
# starting "Error: " like every other failure the model reads.
CANCELLED_TEXT = "Error: cancelled by the caller."


@dataclass(frozen=True)
class Event:
    """One thing that happened. `data` holds only JSON-safe values."""

    type: str
    data: dict[str, Any]
    t: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        """Flat and JSON-ready: {"type": ..., **data}."""
        return {"type": self.type, **self.data}


Listener = Callable[[Event], None]


class EventBus:
    """Who wants to hear about this agent's work, plus its cancel flag."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # A tuple, replaced on every change (copy on write): emit() reads it
        # without the lock and can never see a list mid-edit.
        self._listeners: tuple[Listener, ...] = ()
        # Set by whoever cancels the run. The native loop checks it before
        # every model call; observed() checks it before every tool call, which
        # is the only point a framework engine can be stopped at all.
        self.stop = threading.Event()
        # Facts about the run in progress that a tool may need to read. The
        # A2A server puts the incoming hop count and caller chain here, and
        # the A2A client tool reads them when it calls the next agent.
        self.context: dict[str, Any] = {}

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Add a listener. Returns the function that removes it again."""
        with self._lock:
            self._listeners = self._listeners + (listener,)

        def unsubscribe() -> None:
            with self._lock:
                self._listeners = tuple(
                    x for x in self._listeners if x is not listener)
        return unsubscribe

    def emit(self, type_: str, **data: Any) -> None:
        """Tell every listener. Never raises."""
        listeners = self._listeners
        if not listeners:
            return
        event = Event(type_, data)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                # A broken UI or a dead remote caller must not take the
                # agent down with it.
                pass


def observed(name: str, function: Callable, bus: EventBus) -> Callable:
    """`function`, plus tool events and a cancel check.

    The result object passes through UNCHANGED (the native loop reads the
    list of dicts search_docs returns to collect citations), and exceptions
    still reach the engine, which turns them into "Error: ..." text as it
    always has.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if bus.stop.is_set():
            return CANCELLED_TEXT
        call_id = uuid.uuid4().hex[:8]
        bus.emit(TOOL_STARTED, call_id=call_id, name=name,
                 arguments=dict(kwargs))
        started = time.monotonic()
        status, result_chars = "error", None
        try:
            result = function(*args, **kwargs)
            status, result_chars = "ok", len(str(result))
            return result
        finally:
            bus.emit(TOOL_ENDED, call_id=call_id, name=name, status=status,
                     duration_s=round(time.monotonic() - started, 3),
                     result_chars=result_chars)

    return wrapper
