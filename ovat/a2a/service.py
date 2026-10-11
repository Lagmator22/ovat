# ovat/a2a/service.py
"""Running remote tasks against built OVAT agents. No SDK in here.

Note to myself: this is the part of the server that is OVAT's own logic:
admission, one agent per conversation, one model at a time, cancel,
timeouts, sources, the audit log. The SDK only moves bytes. So this file
talks to the outside through `Reporter`, an abstract interface (in C++, a
pure virtual class) that sdk/server.py implements on top of the SDK's
TaskUpdater, and that tests implement with a list.

Threads, because the engines are synchronous: each task's agent.run()
happens on a worker thread (asyncio.to_thread), while the event loop keeps
serving other callers. A thread from to_thread has no running event loop,
so the LlamaIndex and Agents SDK engines, which start their own, work there
unchanged.
"""
import asyncio
import logging
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Protocol

from ovat.a2a.audit import AuditLog
from ovat.a2a.policy import Admission, Ask
from ovat.agent.events import (REMOTE_STATUS, TOOL_ENDED, TOOL_STARTED,
                               Event)
from ovat.agent.factory import close_agent
from ovat.config.workflow import A2AServerConfig

log = logging.getLogger(__name__)


class Reporter(Protocol):
    """Where one task's progress and result go. One per task.

    Call order: rejected() alone, or accepted() then any number of
    working() then exactly one of failed() or answer() + completed().
    """

    async def accepted(self) -> None: ...
    async def working(self, text: str, event: dict | None = None) -> None: ...
    async def answer(self, text: str, sources: list[str]) -> None: ...
    async def completed(self, trace: dict) -> None: ...
    async def failed(self, text: str, trace: dict | None = None) -> None: ...
    async def rejected(self, text: str) -> None: ...


@dataclass
class Task:
    """One incoming message, in OVAT's terms."""

    task_id: str
    context_id: str
    caller: str
    text: str
    hops: int = 0
    chain: list[str] = field(default_factory=list)


def describe(event: Event) -> str | None:
    """The one line a remote caller sees for an event, or None to skip it."""
    d = event.data
    if event.type == TOOL_STARTED:
        return f"calling {d.get('name')}"
    if event.type == TOOL_ENDED:
        if d.get("status") == "ok":
            return f"{d.get('name')} done in {d.get('duration_s')} s"
        return f"{d.get('name')} {d.get('status')}"
    if event.type == REMOTE_STATUS:
        return f"{d.get('agent')}: {d.get('text')}"
    return None


def public_event(event: Event) -> dict:
    """The event as sent to a remote caller. Tool ARGUMENTS stay here:
    they can hold the local user's paths and data."""
    out = event.to_dict()
    out.pop("arguments", None)
    out.pop("t", None)
    return out


def shape_sources(paths: list[str], mode: str) -> list[str]:
    """Citations as a2a.server.expose_sources says: names, full or none.

    Split on both separators: a Windows path read on any machine is still a
    path, and `C:\\Users\\devcloud\\hr\\leave.md` names the user.
    """
    if mode == "none":
        return []
    if mode == "full":
        return list(paths)
    names = []
    for p in paths:
        name = re.split(r"[\\/]", p)[-1]
        if name and name not in names:
            names.append(name)
    return names


def _sources_of(agent) -> list[str]:
    """Source paths from the native loop's trace; [] for the frameworks,
    which do not record them (absent, not empty: nothing is claimed)."""
    found = []
    for turn in (getattr(agent, "last_trace", None) or {}).get("turns", []):
        for call in turn.get("tool_calls", []):
            for src in call.get("sources") or []:
                if src not in found:
                    found.append(src)
    return found


@dataclass
class _Slot:
    agent: object
    turns: int = 0
    busy: bool = False


class AgentPool:
    """One built agent per (caller, conversation), least recently used out.

    Keyed by caller AND context id: a context id is just a string the
    caller sends, and two callers must never share a memory because one
    guessed the other's id.
    """

    def __init__(self, build: Callable[[], object], size: int,
                 close: Callable = close_agent):
        self._build, self._size, self._close = build, size, close
        self._slots: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def turns(self, key) -> int:
        with self._lock:
            slot = self._slots.get(key)
            return slot.turns if slot else 0

    def acquire(self, key) -> _Slot:
        """The slot for `key`, building its agent on first use. Called on a
        worker thread: building can load an embedder from disk."""
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                self._evict_one_if_full()
                slot = _Slot(self._build())
                self._slots[key] = slot
            self._slots.move_to_end(key)
            slot.busy = True
            return slot

    def release(self, slot: _Slot, counted: bool) -> None:
        with self._lock:
            slot.busy = False
            if counted:
                slot.turns += 1

    def _evict_one_if_full(self) -> None:
        if len(self._slots) < self._size:
            return
        for key, slot in self._slots.items():       # oldest first
            if not slot.busy:
                del self._slots[key]
                self._close(slot.agent)
                return

    def __len__(self) -> int:
        return len(self._slots)

    def close(self) -> None:
        with self._lock:
            slots, self._slots = list(self._slots.values()), OrderedDict()
        for slot in slots:
            self._close(slot.agent)


@dataclass
class _Running:
    stop: threading.Event = field(default_factory=threading.Event)
    agent: object = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class A2AService:
    """Takes remote tasks and runs them, one model call at a time."""

    def __init__(self, build: Callable[[], object], settings: A2AServerConfig,
                 *, server_id: str | None = None,
                 admission: Admission | None = None,
                 audit: AuditLog | None = None,
                 close: Callable = close_agent):
        self.s = settings
        # Random per process: only used to spot a request that has already
        # passed through this server (a cycle between agents).
        self.server_id = server_id or uuid.uuid4().hex[:12]
        self._admission = admission or Admission(settings, self.server_id)
        self._audit = audit
        self._pool = AgentPool(build, settings.max_contexts, close)
        # One model at a time. A threading lock, held by the WORKER thread,
        # because a cancelled task's thread keeps using OVMS until its
        # current request ends; an asyncio lock would be released at the
        # cancel and let the next task start on a busy GPU.
        self._gate = threading.Lock()
        self._counts_lock = threading.Lock()
        self._waiting = 0
        self._running: dict[str, _Running] = {}
        self.stats = {"done": 0, "failed": 0, "rejected": 0, "canceled": 0}

    # ---- the public surface the SDK adapter calls -------------------------

    async def run(self, task: Task, reporter: Reporter) -> None:
        """Run one task to a final state. Never raises, except the
        CancelledError of a cancelled task (the adapter reports that)."""
        key = (task.caller, task.context_id)
        refusal = self._admission.check(Ask(
            caller=task.caller, text=task.text, hops=task.hops,
            chain=list(task.chain), queued=self._waiting,
            turns=self._pool.turns(key)))
        if refusal:
            self.stats["rejected"] += 1
            await reporter.rejected(refusal)
            self._record(task, "rejected", refusal)
            return

        running = self._running[task.task_id] = _Running()
        try:
            await reporter.accepted()
            await self._run_admitted(task, key, running, reporter)
        except asyncio.CancelledError:
            self.cancel(task.task_id)
            self.stats["canceled"] += 1
            self._record(task, "canceled", None)
            raise
        except Exception:
            # Never the exception text: it can carry paths and names from
            # this machine. The full traceback goes to the server's log.
            log.exception("A2A task %s failed", task.task_id)
            self.stats["failed"] += 1
            sentence = ("The agent could not answer because of an error on "
                        "the serving machine. Its log has the details.")
            try:
                await reporter.failed(sentence)
            except Exception:
                pass
            self._record(task, "failed", sentence)
        finally:
            self._running.pop(task.task_id, None)

    def cancel(self, task_id: str) -> bool:
        """Ask a task to stop. False if it is not running here."""
        running = self._running.get(task_id)
        if running is None:
            return False
        with running.lock:
            running.stop.set()
            if running.agent is not None:
                running.agent.events.stop.set()
        return True

    def status(self) -> dict:
        return {"active": len(self._running) - self._waiting,
                "queued": self._waiting, "conversations": len(self._pool),
                **self.stats}

    def close(self) -> None:
        self._pool.close()

    # ---- internals --------------------------------------------------------

    async def _run_admitted(self, task, key, running, reporter) -> None:
        loop = asyncio.get_running_loop()
        tools_used: list[str] = []

        def on_event(event: Event) -> None:          # on the WORKER thread
            if event.type == TOOL_STARTED:
                tools_used.append(event.data.get("name"))
            text = describe(event)
            if text is None:
                return
            future = asyncio.run_coroutine_threadsafe(
                reporter.working(text, public_event(event)), loop)
            try:
                future.result(timeout=5)
            except Exception:
                pass             # a slow or gone caller must not stall the run

        with self._counts_lock:
            ahead = self._waiting + (1 if self._gate.locked() else 0)
            self._waiting += 1
        if ahead:
            await reporter.working(
                f"waiting for the local model ({ahead} ahead)",
                {"type": "queued", "ahead": ahead})

        def work():
            slot = None
            counted = False
            with self._gate:
                with self._counts_lock:
                    self._waiting -= 1
                if running.stop.is_set():
                    return None, None
                slot = self._pool.acquire(key)
                agent = slot.agent
                with running.lock:
                    running.agent = agent
                    agent.events.stop.clear()
                    if running.stop.is_set():
                        agent.events.stop.set()
                agent.events.context = {
                    "hops": task.hops + 1,
                    "chain": [*task.chain, self.server_id]}
                unsubscribe = agent.events.subscribe(on_event)
                try:
                    answer = agent.run(task.text)
                    counted = True
                    return agent, answer
                finally:
                    unsubscribe()
                    agent.events.context = {}
                    with running.lock:
                        running.agent = None
                    self._pool.release(slot, counted)

        worker = asyncio.ensure_future(asyncio.to_thread(work))
        try:
            agent, answer = await asyncio.wait_for(
                asyncio.shield(worker), self.s.task_timeout_s)
        except asyncio.TimeoutError:
            self.cancel(task.task_id)
            self.stats["failed"] += 1
            sentence = (f"The agent did not finish within "
                        f"{self.s.task_timeout_s:g} s "
                        f"(a2a.server.task_timeout_s), so the task was "
                        f"stopped.")
            await reporter.failed(sentence)
            self._record(task, "failed", sentence, tools_used)
            return

        if agent is None or running.stop.is_set():
            return                    # cancelled; the adapter says CANCELED

        totals = (getattr(agent, "last_trace", None) or {}).get("totals") or {}
        failed = bool(totals.get("failed")
                      or getattr(agent, "last_failed", False))
        if failed:
            self.stats["failed"] += 1
            await reporter.failed(answer, totals)
            self._record(task, "failed", answer, tools_used, totals=totals)
            return
        sources = shape_sources(_sources_of(agent), self.s.expose_sources)
        await reporter.answer(answer, sources)
        await reporter.completed(totals)
        self.stats["done"] += 1
        self._record(task, "completed", answer, tools_used, sources, totals)

    def _record(self, task: Task, state: str, answer: str | None,
                tools: list[str] | None = None,
                sources: list[str] | None = None,
                totals: dict | None = None) -> None:
        if self._audit is None:
            return
        self._audit.write({
            "caller": task.caller, "task_id": task.task_id,
            "context_id": task.context_id, "state": state,
            "question": task.text, "answer": answer,
            "tools": tools or [], "sources": sources or [],
            "totals": totals or {}, "hops": task.hops,
        })
