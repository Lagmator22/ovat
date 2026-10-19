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
from ovat.agent.events import TOOL_STARTED, Event, describe
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
    async def completed(self, totals: dict) -> None: ...
    async def failed(self, text: str, totals: dict | None = None) -> None: ...
    async def rejected(self, text: str) -> None: ...
    async def input_required(self, call_id: str, tool: str, args: dict) -> None: ...


@dataclass
class Task:
    """One incoming message, in OVAT's terms."""

    task_id: str
    context_id: str
    caller: str
    text: str
    hops: int = 0
    chain: list[str] = field(default_factory=list)
    data: dict | None = None
    approval_capable: bool = False


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
    users: int = 0          # tasks holding or waiting for this agent
    # One run per conversation at a time. Taken BEFORE the model gate: a run
    # paused for a person's approval gives the gate back but keeps this, so
    # a second message in the same conversation cannot run on the same
    # agent meanwhile.
    lock: threading.Lock = field(default_factory=threading.Lock)


class PoolFull(RuntimeError):
    """Every conversation slot is in use, so a new one cannot be made."""


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

    def has_room(self, key) -> bool:
        with self._lock:
            if key in self._slots or len(self._slots) < self._size:
                return True
            return any(not slot.users for slot in self._slots.values())

    def capacity_refusal(self) -> str:
        return (f"all {self._size} conversations are busy "
                f"(a2a.server.max_contexts)")

    def acquire(self, key) -> _Slot:
        """The slot for `key`, building its agent on first use. Called on a
        worker thread: building can load an embedder from disk."""
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                if not self._make_room():
                    raise PoolFull(
                        f"all {self._size} conversations are busy "
                        f"(a2a.server.max_contexts)")
                slot = _Slot(self._build())
                self._slots[key] = slot
            self._slots.move_to_end(key)
            slot.users += 1
            return slot

    def release(self, slot: _Slot, counted: bool) -> None:
        with self._lock:
            slot.users -= 1
            if counted:
                slot.turns += 1

    def _make_room(self) -> bool:
        """True when one more slot fits: there is space, or the oldest idle
        conversation was dropped. False when all of them are in use."""
        if len(self._slots) < self._size:
            return True
        for key, slot in self._slots.items():       # oldest first
            if not slot.users:
                del self._slots[key]
                self._close(slot.agent)
                return True
        return False

    def __len__(self) -> int:
        return len(self._slots)

    def close(self) -> None:
        with self._lock:
            slots, self._slots = list(self._slots.values()), OrderedDict()
        for slot in slots:
            self._close(slot.agent)


@dataclass
class _PendingApproval:
    task_id: str
    context_id: str
    caller: str
    call_id: str
    tool: str
    args: dict
    event: threading.Event = field(default_factory=threading.Event)
    allowed: bool = False
    answered: bool = False      # by the caller, or denied by a cancel
    expired: bool = False       # nobody answered in time


@dataclass
class _Running:
    stop: threading.Event = field(default_factory=threading.Event)
    agent: object = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    worker_task: asyncio.Task = None
    current_reporter: Reporter = None
    pending_approval: _PendingApproval | None = None
    approval_needed_event: asyncio.Event = field(default_factory=asyncio.Event)
    tools_used: list[str] = field(default_factory=list)
    # False while the task is paused for an approval: its request has ended
    # and nothing awaits the worker, so _settle() has to finish it.
    awaited: bool = True
    expired: bool = False


# Outcomes a caller may send for a tool-call confirmation. The server keeps
# no "always": each answer allows or refuses this one call.
_ALLOW = {"proceed_once", "proceed_always"}


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
        if task.task_id in self._running or (task.data or {}).get("callId"):
            await self._continue(task, reporter)
            return

        key = (task.caller, task.context_id)
        refusal = self._admission.check(Ask(
            caller=task.caller, text=task.text, hops=task.hops,
            chain=list(task.chain), queued=self._waiting,
            turns=self._pool.turns(key)))
        if not refusal and not self._pool.has_room(key):
            refusal = self._pool.capacity_refusal()
        if refusal:
            self.stats["rejected"] += 1
            await reporter.rejected(refusal)
            self._record(task, "rejected", refusal)
            return

        running = self._running[task.task_id] = _Running(current_reporter=reporter)

        async def start():
            await reporter.accepted()
            await self._run_admitted(task, key, running, reporter)
        await self._drive(task, running, reporter, start())

    async def _drive(self, task: Task, running: _Running, reporter: Reporter,
                     step) -> None:
        """Await `step` (a first run, or a run resumed after an approval) and
        turn whatever goes wrong into a final state for the caller."""
        try:
            await step
        except asyncio.CancelledError:
            self.cancel(task.task_id)
            self.stats["canceled"] += 1
            self._record(task, "canceled", None)
            raise
        except PoolFull as exc:
            self.stats["rejected"] += 1
            sentence = str(exc)
            try:
                await reporter.rejected(sentence)
            except Exception:
                pass
            self._record(task, "rejected", sentence)
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
            if running.awaited:
                self._running.pop(task.task_id, None)

    def cancel(self, task_id: str) -> bool:
        """Ask a task to stop. False if it is not running here."""
        running = self._running.get(task_id)
        if running is None:
            return False
        with running.lock:
            running.stop.set()
            pending = running.pending_approval
            if pending is not None and not pending.expired:
                pending.answered, pending.allowed = True, False
                pending.event.set()
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

        def on_event(event: Event) -> None:          # on the WORKER thread
            if event.type == TOOL_STARTED:
                running.tools_used.append(event.data.get("name"))
            text = describe(event)
            if text is None:
                return
            rep = running.current_reporter
            if rep is None:
                return
            future = asyncio.run_coroutine_threadsafe(
                rep.working(text, public_event(event)), loop)
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

        def approver(name: str, arguments: dict, reason: str = "") -> bool:
            """On the WORKER thread, holding the model gate."""
            pend = _PendingApproval(
                task_id=task.task_id, context_id=task.context_id,
                caller=task.caller, call_id=uuid.uuid4().hex[:8],
                tool=name, args=arguments)
            with running.lock:
                if running.stop.is_set():
                    return False
                running.pending_approval = pend
            rep = running.current_reporter
            if rep is not None:
                fut = asyncio.run_coroutine_threadsafe(
                    rep.input_required(pend.call_id, name, arguments), loop)
                try:
                    fut.result(timeout=5)
                except Exception:
                    pass
            loop.call_soon_threadsafe(running.approval_needed_event.set)
            # Nobody uses the model while a person decides: give the gate to
            # the other callers and take it back for the rest of this run.
            self._gate.release()
            try:
                pend.event.wait(timeout=self.s.task_timeout_s)
            finally:
                self._gate.acquire()
            with running.lock:
                if not pend.answered:
                    pend.expired = running.expired = True
                    running.pending_approval = None
                    running.stop.set()
                    if running.agent is not None:
                        running.agent.events.stop.set()
                    return False
            return pend.allowed and not running.stop.is_set()

        def work():
            slot, counted, gated = None, False, False
            try:
                slot = self._pool.acquire(key)   # builds on first use
                with slot.lock, self._gate:
                    gated = True
                    with self._counts_lock:
                        self._waiting -= 1
                    if running.stop.is_set():
                        return None, None
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
                    old_approver = getattr(agent.events, "approver", None)
                    if task.approval_capable and self.s.allow_remote_approvals:
                        agent.events.approver = approver
                    try:
                        answer = agent.run(task.text)
                        counted = True
                        return agent, answer
                    finally:
                        agent.events.approver = old_approver
                        unsubscribe()
                        agent.events.context = {}
                        with running.lock:
                            running.agent = None
            finally:
                if not gated:
                    with self._counts_lock:
                        self._waiting -= 1
                if slot is not None:
                    self._pool.release(slot, counted)

        running.worker_task = asyncio.ensure_future(asyncio.to_thread(work))
        running.worker_task.add_done_callback(
            lambda _f: None if running.awaited
            else asyncio.ensure_future(self._settle(task, running)))
        await self._wait_or_yield(task, running, reporter)

    async def _wait_or_yield(self, task, running, reporter) -> None:
        approval_waiter = asyncio.create_task(running.approval_needed_event.wait())
        try:
            done, _ = await asyncio.wait(
                [running.worker_task, approval_waiter],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=self.s.task_timeout_s)
        finally:
            approval_waiter.cancel()

        if not done:
            self.cancel(task.task_id)
            self.stats["failed"] += 1
            sentence = (f"The agent did not finish within "
                        f"{self.s.task_timeout_s:g} s "
                        f"(a2a.server.task_timeout_s), so the task was "
                        f"stopped.")
            await reporter.failed(sentence)
            self._record(task, "failed", sentence, running.tools_used)
            return

        if running.approval_needed_event.is_set():
            running.approval_needed_event.clear()
            # Paused for a person: this request ends here with
            # input-required, and the caller's answer arrives as a new
            # message (_continue). Until then nothing awaits the worker.
            running.awaited = False
            if running.worker_task.done():
                await self._settle(task, running)
            return

        if running.worker_task.cancelled():
            return

        agent, answer = await running.worker_task
        if agent is None or running.stop.is_set():
            return                    # cancelled; the adapter says CANCELED

        totals = (getattr(agent, "last_trace", None) or {}).get("totals") or {}
        failed = bool(totals.get("failed")
                      or getattr(agent, "last_failed", False))
        if failed:
            self.stats["failed"] += 1
            await reporter.failed(answer, totals)
            self._record(task, "failed", answer, running.tools_used, totals=totals)
            return
        sources = shape_sources(_sources_of(agent), self.s.expose_sources)
        await reporter.answer(answer, sources)
        await reporter.completed(totals)
        self.stats["done"] += 1
        self._record(task, "completed", answer, running.tools_used, sources, totals)

    async def _continue(self, task: Task, reporter: Reporter) -> None:
        """A message for a task paused on an approval: resume the run and
        report it on this request. Only a matching "proceed" allows the
        tool; any other message for the task is a no. A message for a task
        with no open request is refused, never run as a new question."""
        running = self._running.get(task.task_id)
        pending = running.pending_approval if running else None
        refusal = None
        if pending is None:
            refusal = ("This task has no open approval request. It may have "
                       "expired or been answered already.")
        else:
            data = task.data or {}
            allowed = (data.get("callId") == pending.call_id
                       and data.get("outcome") in _ALLOW
                       and (pending.caller, pending.context_id)
                       == (task.caller, task.context_id))
            with running.lock:
                if pending.answered or pending.expired:
                    refusal = "That approval request is already closed."
                else:
                    pending.answered, pending.allowed = True, allowed
                    running.pending_approval = None
                    running.current_reporter = reporter
                    running.awaited = True
        if refusal is not None:
            self.stats["rejected"] += 1
            await reporter.rejected(refusal)
            return
        pending.event.set()
        await self._drive(task, running, reporter,
                          self._wait_or_yield(task, running, reporter))

    async def _settle(self, task: Task, running: _Running) -> None:
        """Finish a run that ended while paused for an approval: nobody
        answered in time, or it was cancelled. Its request is gone, so the
        last reporter is told on a best-effort basis."""
        if self._running.pop(task.task_id, None) is None:
            return
        if running.expired:
            sentence = (f"Nobody answered the approval request within "
                        f"{self.s.task_timeout_s:g} s "
                        f"(a2a.server.task_timeout_s), so the tool was "
                        f"refused and the task stopped.")
            self.stats["failed"] += 1
            try:
                await running.current_reporter.failed(sentence)
            except Exception:
                pass
            self._record(task, "failed", sentence, running.tools_used)
        else:
            self.stats["canceled"] += 1
            self._record(task, "canceled", None, running.tools_used)

    def _record(self, task: Task, state: str, answer: str | None,
                tools: list[str] | None = None,
                sources: list[str] | None = None,
                totals: dict | None = None) -> None:
        if self._audit is None:
            return
        question = (task.text or "")[:self.s.max_message_chars]
        self._audit.write({
            "caller": task.caller, "task_id": task.task_id,
            "context_id": task.context_id, "state": state,
            "question": question, "answer": answer,
            "tools": tools or [], "sources": sources or [],
            "totals": totals or {}, "hops": task.hops,
        })
