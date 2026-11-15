# ovat/agent/events.py
"""In-process agent event bus and uniform tool observation wrapper.

Provides real-time execution visibility (tool calls, status updates, sources)
for front ends and A2A servers without changing engine implementations.

Components:
- ``EventBus``: Thread-safe observer list plus shared cancellation flag
  (``stop``) and per-run ``context`` metadata (such as A2A hop counts).
- ``observed()``: Wraps ``tools[name]["function"]`` uniformly across all four
  engines (native, LangChain, LlamaIndex, OpenAI Agents SDK) to emit
  ``tool_started`` / ``tool_ended`` events, check permissions, and respect ``stop``.

Invariants:
- Standard library only so CLI, TUI, and A2A code do not pull in optional deps.
- Listener exceptions are suppressed so observers never break agent execution.
- Listeners run on the agent worker thread; UI consumers must marshal updates
  to their own event loop (for example via ``post_message`` in Textual).
"""
import contextvars
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
#   tool_ended    call_id, name, status ("ok" | "error" | "denied"),
#                 duration_s, result_chars, and `display` (a dict, only
#                 for tools that return structured output: the coding tools)
#   remote_status agent, state, text      (a remote A2A agent's progress)
#   subagent_status agent, text           (a delegated subagent's progress)
#   subagent_started call_id, agent, kind, task, model, endpoint_host
#   subagent_event   call_id, agent, event (child event dict)
#   subagent_done    call_id, agent, answer, status, duration_s, totals
#   answer_delta  text                    (streamed answer chunk)
#   reasoning_delta text                  (streamed reasoning chunk)
#   steer_taken   text, index             (the native loop took a steering
#                                          message; index is where it sits
#                                          in the model's memory)
TOOL_STARTED = "tool_started"
TOOL_ENDED = "tool_ended"
REMOTE_STATUS = "remote_status"
SUBAGENT_STATUS = "subagent_status"
SUBAGENT_STARTED = "subagent_started"
SUBAGENT_EVENT = "subagent_event"
SUBAGENT_DONE = "subagent_done"
ANSWER_DELTA = "answer_delta"
REASONING_DELTA = "reasoning_delta"
STEER_TAKEN = "steer_taken"

# Context variables for the active tool call and running agent/subagent name.
# Safe across threads and async tasks; reset in finally.
CURRENT_CALL_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "CURRENT_CALL_ID", default="")
# The subagent a tool call runs under, as (name, the delegate call's id):
# an approval names the agent and blocks exactly that child, even when two
# runs of the same subagent are under way.
CURRENT_SUBAGENT: contextvars.ContextVar[tuple[str, str]] = \
    contextvars.ContextVar("CURRENT_SUBAGENT", default=("", ""))


class CombinedStopEvent(threading.Event):
    """A stop event that is set if either itself or its parent is set.

    Allows cancelling an individual child agent without cancelling the parent,
    while still ensuring cancelling the parent cancels all children.
    """

    def __init__(self, parent: threading.Event | None = None) -> None:
        super().__init__()
        self._parent = parent

    def is_set(self) -> bool:
        return super().is_set() or (bool(self._parent and self._parent.is_set()))

    def wait(self, timeout: float | None = None) -> bool:
        # threading.Event.wait() reads its own flag, not is_set(), so it
        # would sleep through the parent being set. Wait in short slices.
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            left = 0.05 if deadline is None else deadline - time.monotonic()
            if left <= 0:
                return False
            super().wait(min(left, 0.05))
        return True


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
        # Asks a person whether a `permission: ask` tool may run:
        # approver(name, arguments, reason) -> bool, may block. None means
        # nobody can be asked, so `ask` is a no (see agent/permissions.py).
        self.approver: Callable[[str, dict, str], bool] | None = None
        # Asks a person a question for the ask_user tool:
        # asker(question, choices) -> answer text, or None if nobody
        # answered. May block. None here means there is no person to ask and
        # the tool says so at once (see agent/questions.py).
        self.asker: Callable[[str, list[str]], str | None] | None = None

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


def describe(event: Event) -> str | None:
    """One plain line for an event ("calling search_docs"), or None.

    Shared by everything that shows another agent's progress as text: the
    A2A server's WORKING updates and a subagent's status lines.
    """
    d = event.data
    if event.type == TOOL_STARTED:
        return f"calling {d.get('name')}"
    if event.type == TOOL_ENDED:
        if d.get("status") == "ok":
            return f"{d.get('name')} done in {d.get('duration_s')} s"
        return f"{d.get('name')} {d.get('status')}"
    if event.type in (REMOTE_STATUS, SUBAGENT_STATUS):
        return f"{d.get('agent')}: {d.get('text')}"
    return None


def observed(name: str, function: Callable, bus: EventBus,
             permission: str = "allow") -> Callable:
    """`function`, plus tool events, a cancel check and its permission.

    The result object passes through UNCHANGED (the native loop reads the
    list of dicts search_docs returns to collect citations), and exceptions
    still reach the engine, which turns them into "Error: ..." text as it
    always has. A tool that may not run returns the sentence saying why,
    which the model reads like any other tool result.
    """
    from ovat.agent.permissions import ALLOW, ASK, gate
    from ovat.tools.project import is_sensitive_path

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if bus.stop.is_set():
            return CANCELLED_TEXT
        call_id = uuid.uuid4().hex[:8]
        token_call_id = CURRENT_CALL_ID.set(call_id)
        try:
            bus.emit(TOOL_STARTED, call_id=call_id, name=name,
                     arguments=dict(kwargs))
            effective_perm = permission
            reason = ""
            file_arg = kwargs.get("file_path")
            if name in ("write_file", "edit") and is_sensitive_path(file_arg):
                if effective_perm == ALLOW:
                    effective_perm = ASK
                reason = f"modifying sensitive file {file_arg}"
            elif name == "read_file" and is_sensitive_path(
                    file_arg, for_read=True):
                if effective_perm == ALLOW:
                    effective_perm = ASK
                reason = f"reading sensitive file {file_arg}"
            refusal = gate(bus, name, dict(kwargs), effective_perm,
                           reason=reason)
            if refusal is not None:
                bus.emit(TOOL_ENDED, call_id=call_id, name=name, status="denied",
                         duration_s=0.0, result_chars=len(refusal))
                return refusal
            started = time.monotonic()
            status, result_chars, display = "error", None, None
            try:
                result = function(*args, **kwargs)
                status, result_chars = "ok", len(str(result))
                # A tool may hand back a str that also carries `display`, the
                # structured version for a UI (see ovat/tools/project.Shown).
                # The engines still get the same object, which is just a str.
                display = getattr(result, "display", None)
                if display and display.get("failed"):
                    status = "error"
                return result
            finally:
                extra = {"display": display} if display else {}
                bus.emit(TOOL_ENDED, call_id=call_id, name=name, status=status,
                         duration_s=round(time.monotonic() - started, 3),
                         result_chars=result_chars, **extra)
        finally:
            CURRENT_CALL_ID.reset(token_call_id)

    return wrapper
