# ovat/tui/server/projector.py
"""AgentSession reports -> the public session the copied TUI reads.

Note to myself: Vibe's client keeps a mirror of the server's session
(history entries, turns, the queue, callbacks) and rebuilds it from
notifications. Each notification carries an `eventId`; the client accepts
only the next number and asks for a fresh read when one is missing. So the
server must keep the TRUE session itself and number every change it
publishes. That is this class: the true state, plus one `_emit` that numbers
and sends.

It is plain synchronous code that runs on the server's event loop. The
session reports from worker threads; ovat/tui/server/context.py hops each
report onto the loop before it gets here, so nothing in this file needs a
lock.

A history entry is never rewritten in place on the wire. Like Vibe's server,
a change is a small JSON patch (append to the answer, replace the state of a
tool call), and an entry that is COMPLETED is frozen: the client refuses a
patch to one.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from uuid import uuid4

from ovat.agent.events import (SUBAGENT_DONE, SUBAGENT_EVENT,
                               SUBAGENT_STARTED, TOOL_ENDED, Event)
from ovat.tui.server.bridge import Bridge
from ovat.tui.vibe.app_server._model import ProtocolModel
from ovat.tui.vibe.app_server._patch import apply_json_patch, make_json_patch
from ovat.tui.vibe.app_server.events import (HistoryEntryAdded,
                                             HistoryEntryUpdated)
from ovat.tui.vibe.app_server.models import (AgentStatsSnapshot,
                                             BlockedEffectState,
                                             CancelledEffectState,
                                             EffectResultDisplay,
                                             FailedEffectState,
                                             FailedSessionStatus,
                                             IdleSessionStatus,
                                             JsonPatchOperation,
                                             PublicCallbackEntry,
                                             PublicChildSession,
                                             PublicEffectEntry,
                                             PublicEntryGenerationStatus,
                                             PublicError, PublicHistoryEntry,
                                             PublicMessageEntry,
                                             PublicSession,
                                             PublicSessionState, PublicTurn,
                                             PublicTurnQueue,
                                             RunningEffectState,
                                             RunningSessionStatus,
                                             TextContentBlock, TokenUsage,
                                             validate_history_entry)
from ovat.tui.vibe.app_server.protocol import (ChildSessionUpdatedParams,
                                               HistoryEntryAddedParams,
                                               HistoryEntryUpdatedParams,
                                               ServerErrorParams,
                                               ServerWarningParams,
                                               SessionUpdatedParams,
                                               StatsUpdatedParams,
                                               TurnCompletedParams,
                                               TurnQueueUpdatedParams,
                                               TurnStartedParams)

Send = Callable[[str, ProtocolModel], None]

# The longest preview a session list shows (Vibe cuts at 160 too).
PREVIEW_CHARS = 160


def now_ms() -> int:
    return int(time.time() * 1000)


class Projector:
    def __init__(self, session: PublicSession, send: Send) -> None:
        self.session = session
        self.history: list[PublicHistoryEntry] = []
        self.turns: list[PublicTurn] = []
        self.queue = PublicTurnQueue()
        self.event_id = 0
        self.bridge = Bridge(session.id)
        # What the stats report beyond the session's token totals.
        self.tool_calls = {"succeeded": 0, "failed": 0, "rejected": 0,
                           "agreed": 0}
        self.child_sessions: dict[str, PublicChildSession] = {}
        self.child_histories: dict[str, list[PublicHistoryEntry]] = {}
        self.child_bridges: dict[str, Bridge] = {}
        self._send = send

    # ---- the state a request reads ---------------------------------------------

    def state(self, *, history_limit: int | None = 200,
              turns_limit: int | None = 200) -> PublicSessionState:
        """The whole session as a snapshot. A limit of None leaves that part
        out, as Vibe does when a read asks for no history or no turns."""
        history = self.history
        return PublicSessionState(
            event_id=self.event_id,
            session=self.session.model_copy(deep=True),
            history=(None if history_limit is None
                     else [e.model_copy(deep=True)
                           for e in history[-history_limit:]]),
            history_before_cursor=(
                history[-history_limit].id
                if history_limit is not None and len(history) > history_limit
                else None),
            turns=(None if turns_limit is None
                   else [t.model_copy(deep=True)
                         for t in self.turns[-turns_limit:]]),
            active_callbacks=[e.model_copy(deep=True)
                              for e in self.open_callbacks()],
            child_sessions=[s.model_copy(deep=True)
                            for s in self.child_sessions.values()],
            turn_queue=self.queue.model_copy(deep=True))

    def open_callbacks(self) -> list[PublicCallbackEntry]:
        return [e for e in self.history
                if isinstance(e, PublicCallbackEntry)
                and e.state.status == "open"]

    def entry(self, entry_id: str) -> PublicHistoryEntry | None:
        return next((e for e in self.history if e.id == entry_id), None)

    # ---- the one place a change is numbered and sent ---------------------------

    def _emit(self, method: str, params_type: type[ProtocolModel],
              **fields) -> None:
        self.event_id += 1
        self._send(method, params_type(
            event_id=self.event_id, session_id=self.session.id,
            emitted_at=now_ms(), **fields))

    # ---- turns -----------------------------------------------------------------

    def turn_started(self, turn: PublicTurn) -> None:
        self.turns.append(turn)
        self.bridge.turn_id = turn.id
        self._emit("turn/started", TurnStartedParams, turn=turn)

    def turn_completed(self, turn: PublicTurn) -> None:
        for index, known in enumerate(self.turns):
            if known.id == turn.id:
                self.turns[index] = turn
        self.bridge.turn_id = None
        self._emit("turn/completed", TurnCompletedParams, turn=turn)

    def queue_updated(self, queue: PublicTurnQueue) -> None:
        self.queue = queue
        self._emit("turn/queueUpdated", TurnQueueUpdatedParams,
                   queue=queue.model_copy(deep=True))

    # ---- the session itself ----------------------------------------------------

    def update_session(self, operations: list[JsonPatchOperation]) -> None:
        raw = self.session.model_dump(mode="json", by_alias=True)
        self.session = PublicSession.model_validate(
            apply_json_patch(raw, operations))
        self._emit("session/updated", SessionUpdatedParams, patch=operations)

    def change_session(self, **fields) -> None:
        """Set fields on the session (python names) and publish the diff."""
        before = self.session.model_dump(mode="json", by_alias=True)
        after = self.session.model_copy(update=fields).model_dump(
            mode="json", by_alias=True)
        operations = make_json_patch(before, after)
        if operations:
            self.update_session([*operations, JsonPatchOperation(
                op="replace", path="/updatedAt", value=now_ms())])

    def set_status(self, status: dict) -> None:
        self.update_session([
            JsonPatchOperation(op="replace", path="/status", value=status),
            JsonPatchOperation(op="replace", path="/updatedAt",
                               value=now_ms())])

    def stats_updated(self, stats: AgentStatsSnapshot,
                      context_window: int) -> None:
        self.session = self.session.model_copy(
            update={"token_usage": stats.token_usage})
        self._emit("session/statsUpdated", StatsUpdatedParams, stats=stats,
                   context_window=context_window)

    # ---- history entries -------------------------------------------------------

    def add_entry(self, entry: PublicHistoryEntry) -> None:
        if self.entry(entry.id) is not None:
            raise ValueError(f"Duplicate public history entry: {entry.id}")
        self.history.append(entry)
        self._emit("history/entryAdded", HistoryEntryAddedParams,
                   turn_id=entry.turn_id, entry=entry)

    def patch_entry(self, entry_id: str,
                    operations: list[JsonPatchOperation]) -> bool:
        """Change an entry by patch. False when it is gone or already
        COMPLETED (a late report for a finished entry is dropped, not an
        error: the client would refuse the patch anyway)."""
        current = self.entry(entry_id)
        if current is None or (current.generation_status
                               is PublicEntryGenerationStatus.COMPLETED):
            return False
        operations = [*operations, JsonPatchOperation(
            op="replace", path="/updatedAt", value=now_ms())]
        updated = validate_history_entry(apply_json_patch(
            current.model_dump(mode="json", by_alias=True), operations))
        self.history[self.history.index(current)] = updated
        self._emit("history/entryUpdated", HistoryEntryUpdatedParams,
                   turn_id=updated.turn_id, entry_id=entry_id,
                   patch=operations)
        return True

    def complete_effect(self, entry_id: str, state) -> bool:
        return self.patch_entry(entry_id, [
            JsonPatchOperation(op="replace", path="/state",
                               value=state.model_dump(mode="json",
                                                      by_alias=True)),
            JsonPatchOperation(op="replace", path="/generationStatus",
                               value="completed")])

    def add_message(self, entry: PublicMessageEntry) -> None:
        self.add_entry(entry)
        if entry.role == "user" and not self.session.preview:
            text = entry.text.strip()
            if text:
                self.update_session([
                    JsonPatchOperation(op="replace", path="/preview",
                                       value=text[:PREVIEW_CHARS]),
                    JsonPatchOperation(op="replace", path="/updatedAt",
                                       value=now_ms())])

    # ---- the agent's tool, subagent and remote events --------------------------

    def agent_event(self, event: Event, kinds: dict[str, str]) -> None:
        if event.type == SUBAGENT_STARTED:
            call_id = event.data.get("call_id") or ""
            child_id = f"child-{call_id}"
            agent_name = event.data.get("agent") or "subagent"
            kind = event.data.get("kind", "workflow")
            active_turn = self.bridge.turn_id or f"turn-{call_id}"
            task_text = event.data.get("task") or ""
            child_session = PublicChildSession(
                id=child_id,
                name=agent_name,
                agent_type=kind,
                status=RunningSessionStatus(active_turn_id=active_turn),
                created_at=now_ms(),
                updated_at=now_ms(),
            )
            self.child_sessions[child_id] = child_session
            self.child_histories[child_id] = []
            self.child_bridges[child_id] = Bridge(session_id=child_id)
            if task_text:
                stamp = now_ms()
                self.child_histories[child_id].append(PublicMessageEntry(
                    id=str(uuid4()), session_id=child_id,
                    turn_id=active_turn, created_at=stamp, updated_at=stamp,
                    generation_status=PublicEntryGenerationStatus.COMPLETED,
                    role="user", content=[TextContentBlock(text=task_text)],
                    source="turn_start"))
            self._emit("session/childSessionUpdated", ChildSessionUpdatedParams,
                       child_session=child_session)
            return

        if event.type == SUBAGENT_EVENT:
            call_id = event.data.get("call_id") or ""
            child_id = f"child-{call_id}"
            raw_event = event.data.get("event")
            if child_id in self.child_bridges and isinstance(raw_event, dict):
                child_bridge = self.child_bridges[child_id]
                child_ev = Event(raw_event.get("type", ""),
                                 {k: v for k, v in raw_event.items()
                                  if k not in ("type", "t")})
                child_update = child_bridge.convert(child_ev, kinds)
                child_history = self.child_histories.get(child_id, [])
                if isinstance(child_update, HistoryEntryAdded):
                    child_history.append(child_update.entry)
                elif isinstance(child_update, HistoryEntryUpdated):
                    for idx, e in enumerate(child_history):
                        if e.id == child_update.entry.id:
                            child_history[idx] = child_update.entry
                            break
            return

        if event.type == SUBAGENT_DONE:
            call_id = event.data.get("call_id") or ""
            child_id = f"child-{call_id}"
            if child_id in self.child_sessions:
                child = self.child_sessions[child_id]
                status_str = event.data.get("status", "ok")
                status = (FailedSessionStatus() if status_str == "failed"
                          else IdleSessionStatus())
                totals = event.data.get("totals") or {}
                prompt_toks = totals.get("prompt_tokens") or 0
                compl_toks = totals.get("completion_tokens") or 0
                tot_toks = totals.get("total_tokens") or (prompt_toks + compl_toks)
                usage = TokenUsage(input_tokens=prompt_toks,
                                   output_tokens=compl_toks,
                                   total_tokens=tot_toks)
                child = child.model_copy(update={
                    "status": status,
                    "token_usage": usage,
                    "context_usage": usage,
                    "updated_at": now_ms(),
                })
                self.child_sessions[child_id] = child
                self._emit("session/childSessionUpdated", ChildSessionUpdatedParams,
                           child_session=child)
                answer_text = event.data.get("answer") or ""
                if answer_text and child_id in self.child_histories:
                    stamp = now_ms()
                    active_turn = self.bridge.turn_id or f"turn-{call_id}"
                    self.child_histories[child_id].append(PublicMessageEntry(
                        id=str(uuid4()), session_id=child_id,
                        turn_id=active_turn, created_at=stamp, updated_at=stamp,
                        generation_status=PublicEntryGenerationStatus.COMPLETED,
                        role="assistant", content=[TextContentBlock(text=answer_text)],
                        source="agent_response"))
                curr_usage = self.session.token_usage
                base_in = curr_usage.input_tokens if curr_usage is not None else 0
                base_out = curr_usage.output_tokens if curr_usage is not None else 0
                base_tot = curr_usage.total_tokens if curr_usage is not None else 0
                self.session = self.session.model_copy(update={
                    "token_usage": TokenUsage(
                        input_tokens=base_in + prompt_toks,
                        output_tokens=base_out + compl_toks,
                        total_tokens=base_tot + tot_toks,
                    )
                })
            return

        if event.type == TOOL_ENDED:
            status = event.data.get("status")
            key = {"ok": "succeeded", "error": "failed",
                   "denied": "rejected"}.get(status)
            if key:
                self.tool_calls[key] += 1
        update = self.bridge.convert(event, kinds)
        if isinstance(update, HistoryEntryAdded):
            self.add_entry(update.entry)
        elif isinstance(update, HistoryEntryUpdated):
            if update.patch:
                self.patch_entry(update.entry.id, update.patch)
            else:
                self.complete_effect(update.entry.id, update.entry.state)

    # ---- approvals -------------------------------------------------------------

    def open_callback(self, entry: PublicCallbackEntry) -> None:
        """Show the question, and mark the tool call that waits for it."""
        self.add_entry(entry)
        related = self.entry(entry.related_entry_id or "")
        if isinstance(related, PublicEffectEntry) and isinstance(
                related.state, RunningEffectState):
            self.patch_entry(related.id, [JsonPatchOperation(
                op="replace", path="/state",
                value=BlockedEffectState(
                    callback_id=entry.callback_id,
                    output_text=related.state.output_text,
                ).model_dump(mode="json", by_alias=True))])

    def close_callback(self, callback_id: str, state: dict) -> bool:
        """Settle a callback entry with `state` ({"status": "answered", ...}
        or "cancelled") and let its tool call run again."""
        entry = next((e for e in self.history
                      if isinstance(e, PublicCallbackEntry)
                      and e.callback_id == callback_id), None)
        if entry is None:
            return False
        done = self.patch_entry(entry.id, [
            JsonPatchOperation(op="replace", path="/state", value=state),
            JsonPatchOperation(op="replace", path="/generationStatus",
                               value="completed")])
        related = self.entry(entry.related_entry_id or "")
        if isinstance(related, PublicEffectEntry) and isinstance(
                related.state, BlockedEffectState):
            self.patch_entry(related.id, [JsonPatchOperation(
                op="replace", path="/state",
                value=RunningEffectState(
                    output_text=related.state.output_text,
                ).model_dump(mode="json", by_alias=True))])
        return done

    # ---- the end of a turn -----------------------------------------------------

    def finalize_turn(self, turn_id: str, *, cancelled: bool) -> None:
        """Close what the turn left open: a tool call that never reported an
        end, a question nobody answered. Entries outside the turn (a `!`
        command running meanwhile) are not touched."""
        reason = "Turn interrupted" if cancelled else "Turn ended"
        for entry in list(self.history):
            if (entry.turn_id != turn_id or entry.generation_status
                    is PublicEntryGenerationStatus.COMPLETED):
                continue
            if isinstance(entry, PublicEffectEntry):
                display = EffectResultDisplay(success=False, message=reason)
                output = getattr(entry.state, "output_text", "")
                state = (CancelledEffectState(reason=reason,
                                              output_text=output,
                                              display=display)
                         if cancelled else
                         FailedEffectState(error=PublicError(message=reason),
                                           output_text=output,
                                           display=display))
                self.complete_effect(entry.id, state)
            elif isinstance(entry, PublicCallbackEntry):
                self.close_callback(entry.callback_id, {
                    "status": "cancelled", "reason": reason})
            else:
                self.patch_entry(entry.id, [JsonPatchOperation(
                    op="replace", path="/generationStatus",
                    value="completed")])

    # ---- warnings --------------------------------------------------------------

    def warning(self, message: str, code: str | None = None) -> None:
        self._send("warning", ServerWarningParams(
            warning=PublicError(message=message, code=code)))

    def error(self, message: str) -> None:
        self._send("error", ServerErrorParams(
            error=PublicError(message=message)))

    # ---- another conversation ------------------------------------------------

    def adopt(self, session: PublicSession,
              history: list[PublicHistoryEntry],
              turns: list[PublicTurn],
              child_sessions: dict[str, PublicChildSession] | None = None,
              child_histories: dict[str, list[PublicHistoryEntry]] | None = None) -> None:
        """Become another conversation: a new one (/clear), a resumed one, a
        rewound one, a fork.

        Nothing is sent: the method that does this answers with the whole
        new state, and the client adopts it in place of its mirror. The
        numbering carries on, so the client's next event is still the
        next."""
        self.session = session
        self.history = list(history)
        self.turns = list(turns)
        self.child_sessions = dict(child_sessions or {})
        self.child_histories = dict(child_histories or {})
        self.child_bridges = {cid: Bridge(cid) for cid in self.child_sessions}
        self.queue = PublicTurnQueue()
        # A new bridge counts from 1 again, and an entry it made would
        # collide with one the kept history already holds ("msg-1"): a
        # compacted or rewound conversation could not show its next answer.
        used = [int(e.id.rsplit("-", 1)[1]) for e in self.history
                if e.id.rsplit("-", 1)[-1].isdigit()]
        self.bridge = Bridge(session.id, first_id=max(used, default=0) + 1)


