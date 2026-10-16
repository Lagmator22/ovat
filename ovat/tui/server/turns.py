# ovat/tui/server/turns.py
"""Turns: starting one, the queue behind it, and how one ends.

Note to myself: the agent runs ONE turn at a time (AgentSession.ask refuses
a second). Vibe's TUI sends every prompt through a queue (`session/turn/
enqueue`): an idle server promotes the item at once, a busy one keeps it. So
this file owns both halves:

- the turn that is running (`active`), from the request to the moment the
  session reports TurnDone (or the moment it could never start), and
- the queue of prompts that came in while it ran.

The order of events matches Vibe's server, which the client's bookkeeping
depends on: the response first, then queueUpdated (the item is there), then
queueUpdated (it was promoted), then turn/started, the user's message, the
agent's entries, the stats, and turn/completed last.

A prompt that arrives before the agent has finished building (loading a model
takes a while) is accepted: the turn starts, waits for the build, and then
asks. If the build failed the turn fails with that sentence.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from ovat.agentic import session as s
from ovat.text import THINK_CLOSE, THINK_OPEN
from ovat.tui.server.projector import now_ms
from ovat.tui.server.rpc import Reply, RpcError
from ovat.tui.vibe.app_server.models import (PublicEntryGenerationStatus,
                                             PublicError, PublicMessageEntry,
                                             PublicQueuedTurn, PublicTurn,
                                             PublicTurnQueue,
                                             PublicTurnStatus,
                                             PublicTurnStopReason,
                                             SessionTextContentBlock,
                                             TextContentBlock,
                                             TurnUserInputEntry)
from ovat.tui.vibe.app_server.protocol import (ProtocolErrorCode,
                                               TurnEnqueueParams,
                                               TurnEnqueueResponse,
                                               TurnStartParams,
                                               TurnStartResponse)

QUEUE_MAX = 32


class BuildFailedError(Exception):
    """The agent could not be built; the message is what the person reads."""


@dataclass
class _Active:
    turn: PublicTurn
    text: str
    entry_id: str
    blocks: list
    injected: bool = False
    task: asyncio.Task | None = None
    cancel_requested: bool = False
    agent_started: bool = False


@dataclass
class _Queued:
    item: PublicQueuedTurn
    params: TurnEnqueueParams
    text: str
    entry_id: str | None


class TurnManager:
    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.active: _Active | None = None
        self.queue: list[_Queued] = []
        self.paused = False
        # idempotency key -> (the params it came with, the queue item it made)
        self._receipts: dict[str, tuple[TurnEnqueueParams, str]] = {}

    @property
    def active_id(self) -> str | None:
        return self.active.turn.id if self.active else None

    @property
    def busy(self) -> bool:
        return self.active is not None

    # ---- starting --------------------------------------------------------------

    def start(self, params: TurnStartParams) -> Reply:
        """`turn/start`: a turn right now. Refused while one runs or while
        prompts wait in the queue (they were first)."""
        ctx = self.ctx
        ctx.require_session(params.session_id)
        if self.active is not None:
            raise RpcError(ProtocolErrorCode.CONFLICT,
                           "A turn is already running")
        if self.queue:
            raise RpcError(ProtocolErrorCode.CONFLICT,
                           "Resume queued turns before starting a new turn")
        text = _text_of(params.message)
        if not text.strip():
            raise RpcError(ProtocolErrorCode.INVALID_PARAMS,
                           "Turn input is empty")
        turn, begin = self._prepare(
            text, params.client_user_message_id,
            [b for b in params.message if isinstance(b, TextContentBlock)],
            queue_item_id=None, injected=params.injected)
        self._warn_if_dropped(params.message)
        return Reply(TurnStartResponse(
            turn=turn, last_event_id=ctx.projector.event_id), after=begin)

    def _prepare(self, text: str, entry_id: str | None, blocks: list, *,
                 queue_item_id: str | None, injected: bool = False
                 ) -> tuple[PublicTurn, Callable[[], None]]:
        ctx = self.ctx
        projector = ctx.projector
        # An id the history already holds (a client that reuses one) would be
        # refused as a duplicate and leave the turn half started.
        if not entry_id or projector.entry(entry_id) is not None:
            entry_id = str(uuid4())
        turn = PublicTurn(
            id=str(uuid4()), session_id=projector.session.id,
            status=PublicTurnStatus.IN_PROGRESS, input_entry_id=entry_id,
            started_at=now_ms(), queue_item_id=queue_item_id)
        active = _Active(turn, text, entry_id,
                         blocks or [TextContentBlock(text=text)], injected)
        self.active = active

        def begin() -> None:
            announced = False
            try:
                projector.turn_started(turn)
                announced = True
                projector.set_status({"type": "running",
                                      "activeTurnId": turn.id})
                stamp = now_ms()
                projector.add_message(PublicMessageEntry(
                    id=entry_id, session_id=projector.session.id,
                    turn_id=turn.id, created_at=stamp, updated_at=stamp,
                    generation_status=PublicEntryGenerationStatus.COMPLETED,
                    role="user", content=active.blocks,
                    source="harness" if injected else "turn_start"))
                active.task = ctx.loop.create_task(self._run(active),
                                                   name="ovat-turn")
            except Exception as exc:
                self._fail_start(active, announced, exc)
        return turn, begin

    def _fail_start(self, active: _Active, announced: bool,
                    exc: Exception) -> None:
        """A turn that could not even begin must not leave the server
        thinking it is busy for good."""
        message = f"Could not start the turn: {type(exc).__name__}: {exc}"
        if announced:
            self._complete(active, PublicTurnStatus.FAILED,
                           error=PublicError(message=message))
        elif self.active is active:
            self.active = None
        self.ctx.report_error(message)

    async def _run(self, active: _Active) -> None:
        ctx = self.ctx
        try:
            await ctx.wait_ready()
            if active.cancel_requested:
                self._complete(active, PublicTurnStatus.INTERRUPTED)
                return
            agent = ctx.agent
            # `@name task` names a subagent only once the workflow (and so
            # its subagents) is known.
            subagent = _subagent_call(active.text, ctx.info)
            if subagent is not None:
                started = agent.ask_subagent(*subagent)
            else:
                started = agent.ask(active.text)
            if started is None:
                reason = ("This workflow has no subagents."
                          if subagent is not None and "delegate" not in (
                              getattr(agent.agent, "tools", None) or {})
                          else "The agent could not start this turn.")
                self._complete(active, PublicTurnStatus.FAILED,
                               error=PublicError(message=reason))
        except asyncio.CancelledError:
            self._complete(active, PublicTurnStatus.INTERRUPTED)
        except BuildFailedError as exc:
            self._complete(active, PublicTurnStatus.FAILED,
                           error=PublicError(message=str(exc)))
        except Exception as exc:
            self._complete(active, PublicTurnStatus.FAILED,
                           error=PublicError(
                               message=f"{type(exc).__name__}: {exc}"))

    def _warn_if_dropped(self, blocks) -> None:
        if any(not isinstance(b, TextContentBlock) for b in blocks):
            self.ctx.projector.warning(
                "Only the text of your message was sent: images and "
                "attached files are not supported in OVAT yet.")

    # ---- the agent reports -----------------------------------------------------

    def agent_started(self) -> None:
        """The session's own TurnStarted. A cancel that came in before the
        worker started would have been cleared by it, so it is repeated."""
        active = self.active
        if active is None:
            return
        active.agent_started = True
        if active.cancel_requested:
            self.ctx.agent.cancel()

    def finish(self, event: s.TurnDone) -> None:
        active = self.active
        if active is None:
            return
        ctx = self.ctx
        projector = ctx.projector
        ctx.record_totals(event)
        if event.cancelled:
            status = PublicTurnStatus.INTERRUPTED
        elif event.failed:
            status = PublicTurnStatus.FAILED
        else:
            status = PublicTurnStatus.COMPLETED
        error = (PublicError(message=event.answer or "The turn failed.")
                 if status is PublicTurnStatus.FAILED else None)
        if status is PublicTurnStatus.COMPLETED:
            try:
                if event.reasoning.strip():
                    projector.add_entry(projector.bridge.reasoning(
                        _untag(event.reasoning)).entry)
                projector.add_entry(projector.bridge.answer(
                    event.answer or "(no answer)").entry)
            except Exception as exc:
                status = PublicTurnStatus.FAILED
                error = PublicError(
                    message=f"Could not show the answer: "
                            f"{type(exc).__name__}: {exc}")
        truncated = bool((event.totals or {}).get("truncated"))
        self._complete(active, status, error=error,
                       stop_reason=(PublicTurnStopReason.LENGTH
                                    if truncated
                                    and status is not
                                    PublicTurnStatus.INTERRUPTED else None),
                       cancelled=event.cancelled)

    def _complete(self, active: _Active, status: PublicTurnStatus, *,
                  error: PublicError | None = None,
                  stop_reason: PublicTurnStopReason | None = None,
                  cancelled: bool | None = None) -> None:
        if self.active is not active:
            return
        ctx = self.ctx
        projector = ctx.projector
        completed = active.turn.model_copy(update={
            "status": status, "completed_at": now_ms(), "error": error,
            "stop_reason": stop_reason})
        try:
            ctx.approvals.cancel_all("Turn ended")
            projector.finalize_turn(
                active.turn.id,
                cancelled=(status is PublicTurnStatus.INTERRUPTED
                           if cancelled is None else cancelled))
            projector.stats_updated(ctx.stats.snapshot(), 0)
            projector.set_status({"type": "idle"})
        except Exception as exc:
            # Whatever went wrong, the turn still ends: a server that stays
            # "busy" for ever is worse than a missing detail.
            ctx.report_error(f"{type(exc).__name__}: {exc}")
        self.active = None
        projector.turn_completed(completed)
        self._after_turn(status)

    def _after_turn(self, status: PublicTurnStatus) -> None:
        # An interrupted turn pauses the queue (the person decides whether
        # the rest still makes sense); any other end moves on.
        if status is PublicTurnStatus.INTERRUPTED:
            if self.queue and not self.paused:
                self.paused = True
                self._emit_queue()
            return
        self.promote_next()

    # ---- interrupting ----------------------------------------------------------

    def interrupt(self, expected_turn_id: str) -> None:
        active = self.active
        if active is None:
            raise RpcError(ProtocolErrorCode.CONFLICT, "No active turn")
        if active.turn.id != expected_turn_id:
            raise RpcError(ProtocolErrorCode.STALE_TURN,
                           f"The active turn is {active.turn.id}")
        self.cancel_active()

    def cancel_active(self) -> None:
        active = self.active
        if active is None:
            return
        active.cancel_requested = True
        self.ctx.agent.cancel()
        self.ctx.approvals.cancel_all("Turn interrupted")
        if active.task is not None and not active.agent_started:
            # Still waiting for the build (or about to ask): stop waiting.
            active.task.cancel()

    # ---- the queue -------------------------------------------------------------

    def enqueue(self, params: TurnEnqueueParams) -> Reply:
        ctx = self.ctx
        ctx.require_session(params.session_id)
        text, entry_id = _queued_text(params)
        key = params.idempotency_key
        if key is not None and key in self._receipts:
            known_params, item_id = self._receipts[key]
            if known_params != params:
                raise RpcError(
                    ProtocolErrorCode.CONFLICT,
                    f"Idempotency key was already used with different input: "
                    f"{key}")
            return Reply(TurnEnqueueResponse(queue_item_id=item_id))
        if len(self.queue) >= QUEUE_MAX:
            raise RpcError(ProtocolErrorCode.CONFLICT,
                           f"Turn queue is full ({QUEUE_MAX} items)")
        item = PublicQueuedTurn(
            id=str(uuid4()), created_at=now_ms(),
            entries=[e.model_copy(deep=True) for e in params.entries])
        self.queue.append(_Queued(item, params.model_copy(deep=True), text,
                                  entry_id))
        if key is not None:
            self._receipts[key] = (params.model_copy(deep=True), item.id)
        return Reply(TurnEnqueueResponse(queue_item_id=item.id),
                     after=self.after_queue_change)

    def replace(self, queue_item_id: str, params: TurnEnqueueParams) -> Reply:
        self.ctx.require_session(params.session_id)
        text, entry_id = _queued_text(params)
        for index, queued in enumerate(self.queue):
            if queued.item.id == queue_item_id:
                break
        else:
            raise RpcError(ProtocolErrorCode.NOT_FOUND,
                           f"Queued turn not found: {queue_item_id}")
        item = queued.item.model_copy(update={
            "entries": [e.model_copy(deep=True) for e in params.entries]})
        self.queue[index] = _Queued(item, params.model_copy(deep=True), text,
                                    entry_id)
        if params.idempotency_key is not None:
            self._receipts[params.idempotency_key] = (
                params.model_copy(deep=True), queue_item_id)
        return Reply(_replace_response(queue_item_id),
                     after=self.after_queue_change)

    def remove(self, queue_item_id: str) -> Reply:
        from ovat.tui.vibe.app_server.protocol import TurnQueueRemoveResponse
        for index, queued in enumerate(self.queue):
            if queued.item.id == queue_item_id:
                del self.queue[index]
                if not self.queue:
                    self.paused = False
                return Reply(TurnQueueRemoveResponse(),
                             after=self.after_queue_change)
        return Reply(TurnQueueRemoveResponse())

    def resume(self) -> Reply:
        from ovat.tui.vibe.app_server.protocol import TurnQueueResumeResponse
        if not self.paused:
            return Reply(TurnQueueResumeResponse())
        self.paused = False
        return Reply(TurnQueueResumeResponse(), after=self.after_queue_change)

    def queue_state(self) -> PublicTurnQueue:
        return PublicTurnQueue(items=[q.item.model_copy(deep=True)
                                      for q in self.queue],
                               paused=self.paused, max_items=QUEUE_MAX)

    def after_queue_change(self) -> None:
        self._emit_queue()
        self.promote_next()

    def _emit_queue(self) -> None:
        self.ctx.projector.queue_updated(self.queue_state())

    def promote_next(self) -> None:
        if self.active is not None or self.paused or not self.queue:
            return
        queued = self.queue.pop(0)
        blocks = [TextContentBlock(text=b.text)
                  for e in queued.item.entries
                  if isinstance(e, TurnUserInputEntry)
                  for b in e.content if isinstance(b, SessionTextContentBlock)]
        _, begin = self._prepare(queued.text, queued.entry_id, blocks,
                                 queue_item_id=queued.item.id)
        self._emit_queue()
        begin()

    # ---- a new conversation / shutdown -----------------------------------------

    def reset(self) -> None:
        self.queue.clear()
        self.paused = False
        self._receipts.clear()

    def close(self) -> None:
        active = self.active
        if active is not None and active.task is not None:
            active.task.cancel()


def _replace_response(queue_item_id: str):
    from ovat.tui.vibe.app_server.protocol import TurnQueueReplaceResponse
    return TurnQueueReplaceResponse(queue_item_id=queue_item_id)


def _text_of(blocks) -> str:
    return "\n\n".join(b.text for b in blocks
                       if isinstance(b, (TextContentBlock,
                                         SessionTextContentBlock))
                       and b.text)


def _queued_text(params: TurnEnqueueParams) -> tuple[str, str | None]:
    """The user's text and its entry id. Context entries and entries with no
    text are refused here, so a promoted item always has something to ask."""
    if any(e.role != "user" for e in params.entries):
        raise RpcError(ProtocolErrorCode.INVALID_PARAMS,
                       "Context entries are not supported in OVAT yet.")
    entry = params.user_entry
    text = _text_of(entry.content) if entry is not None else ""
    if not text.strip():
        raise RpcError(ProtocolErrorCode.INVALID_PARAMS,
                       "Turn input is empty")
    return text, entry.entry_id


def _subagent_call(text: str, info: dict) -> tuple[str, str] | None:
    """`@name task` with the name of one of this workflow's subagents goes
    straight to that subagent, like `/ask name task`. Any other text, and an
    `@` that names a file, is an ordinary prompt."""
    text = text.strip()
    if not text.startswith("@"):
        return None
    head, _, rest = text[1:].partition(" ")
    names = {s["name"] for s in info.get("subagents", [])}
    task = rest.strip()
    if head in names and task:
        return head, task
    return None


def _untag(reasoning: str) -> str:
    """The model's reasoning without its <think> markers (markdown would
    read them as HTML and show nothing)."""
    return THINK_CLOSE.sub("", THINK_OPEN.sub("", reasoning)).strip()
