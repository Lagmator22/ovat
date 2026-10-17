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

from ovat.agent.events import TOOL_ENDED, Event
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
                                             JsonPatchOperation,
                                             PublicCallbackEntry,
                                             PublicEffectEntry,
                                             PublicEntryGenerationStatus,
                                             PublicError, PublicHistoryEntry,
                                             PublicMessageEntry,
                                             PublicSession,
                                             PublicSessionState, PublicTurn,
                                             PublicTurnQueue, RunningEffectState,
                                             validate_history_entry)
from ovat.tui.vibe.app_server.protocol import (HistoryEntryAddedParams,
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
              turns: list[PublicTurn]) -> None:
        """Become another conversation: a new one (/clear), a resumed one, a
        rewound one, a fork.

        Nothing is sent: the method that does this answers with the whole
        new state, and the client adopts it in place of its mirror. The
        numbering carries on, so the client's next event is still the
        next."""
        self.session = session
        self.history = list(history)
        self.turns = list(turns)
        self.queue = PublicTurnQueue()
        self.bridge = Bridge(session.id)

