# ovat/tui/server/approvals.py
"""Approvals: the agent's ApprovalBroker, shown as Vibe callbacks.

Note to myself: when a tool needs a person's yes, the agent's worker thread
BLOCKS inside ApprovalBroker.__call__ until somebody calls
broker.resolve(id, decision). Vibe calls the same moment a "callback": the
server sends the client a `callback/call` request carrying a callback entry,
the person answers, the client sends `callback/result`, and that answer is
what resolves the broker here.

Vibe's choices map onto the broker's three decisions:

    approve                  -> ONCE
    approve for the session  -> SESSION
    approve permanently      -> ALWAYS    (saved in .ovat/permissions.json)
    deny                     -> REFUSE
    cancel the turn          -> REFUSE, and the turn is interrupted

A model-written argument is shown to the person deciding through
`safe.approval_safe`, so it cannot dress up as part of the panel.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter

from ovat.agent.permissions import (ALWAYS, ONCE, REFUSE, SESSION,
                                    ApprovalRequest)
from ovat.tui.server.projector import now_ms
from ovat.tui.server.safe import approval_safe
from ovat.tui.server.rpc import RpcError
from ovat.tui.vibe.app_server._effect_models import GenericEffectDetail
from ovat.tui.vibe.app_server.models import (ApprovalCallbackDetail,
                                             ApprovalCallbackOutput,
                                             ApprovalDecisionType,
                                             BlockedSessionStatus,
                                             CallbackOutput, OpenCallbackState,
                                             PublicCallbackEntry,
                                             PublicEntryGenerationStatus,
                                             RunningSessionStatus)
from ovat.tui.vibe.app_server.protocol import (CallbackCallParams,
                                                CallbackCallResponse,
                                                CallbackResultParams,
                                                ChildSessionUpdatedParams,
                                                ProtocolErrorCode)
from ovat.tui.vibe.utils.tool_presentation import EffectCallDisplay

# What the person is offered. "Permanently" is saved per project by the
# session's broker (ovat/agentic/rules.py), and only an interactive session
# has a broker that reads it back.
CHOICES = [ApprovalDecisionType.APPROVE,
           ApprovalDecisionType.APPROVE_FOR_SESSION,
           ApprovalDecisionType.APPROVE_PERMANENTLY,
           ApprovalDecisionType.DENY]

_DECISIONS = {
    ApprovalDecisionType.APPROVE: ONCE,
    ApprovalDecisionType.APPROVE_FOR_SESSION: SESSION,
    ApprovalDecisionType.APPROVE_PERMANENTLY: ALWAYS,
    ApprovalDecisionType.DENY: REFUSE,
    ApprovalDecisionType.CANCEL_TURN: REFUSE,
}

_OUTPUT = TypeAdapter(CallbackOutput)


@dataclass
class _Open:
    request: ApprovalRequest
    callback_id: str
    answer: dict | None = None        # the output the person gave, once given
    child_id: str | None = None


class Approvals:
    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self._open: dict[str, _Open] = {}
        # Callbacks that ended without an answer (turn cancelled): a late
        # answer is "closed", not "unknown".
        self._closed: set[str] = set()

    # ---- the broker asks -------------------------------------------------------

    def request(self, request: ApprovalRequest) -> None:
        ctx = self._ctx
        projector = ctx.projector
        if projector is None or ctx.rpc is None:
            ctx.agent.broker.resolve(request.id, REFUSE)
            return
        if "approval" not in ctx.capabilities.callback_kinds:
            # Nobody can be asked: that is a no, like a run with no UI.
            ctx.agent.broker.resolve(request.id, REFUSE)
            projector.warning(
                f"{request.tool} needs your approval, and this client cannot "
                f"ask for one. It was not allowed.")
            return
        callback_id = f"approval-{request.id}"
        running = projector.bridge.running(request.tool)
        # The panel gets a generic, sanitised view of the call, never the
        # tool's own card: the arguments are model-written text.
        effect = GenericEffectDetail(
            tool_name=request.tool,
            input=approval_safe(request.arguments) or None,
            display=EffectCallDisplay(
                summary=request.tool,
                status_text=f"Asking about {request.tool}"))
        stamp = now_ms()
        child_id = None
        if request.agent and projector:
            for cid, cs in projector.child_sessions.items():
                if cs.name == request.agent and isinstance(cs.status, RunningSessionStatus):
                    child_id = cid
                    blocked_cs = cs.model_copy(update={
                        "status": BlockedSessionStatus(
                            active_turn_id=ctx.turns.active_id or "",
                            callback_id=callback_id,
                            reason="approval"
                        ),
                        "updated_at": stamp
                    })
                    projector.child_sessions[cid] = blocked_cs
                    projector._emit("session/childSessionUpdated", ChildSessionUpdatedParams,
                                    child_session=blocked_cs)
                    break
        title = (f"Allow subagent {request.agent} to run {request.tool}?"
                 if request.agent else f"Allow {request.tool}?")
        entry = PublicCallbackEntry(
            id=f"callback:{callback_id}", session_id=projector.session.id,
            turn_id=ctx.turns.active_id, created_at=stamp, updated_at=stamp,
            generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
            related_entry_id=running.id if running else None,
            callback_id=callback_id, title=title,
            detail=ApprovalCallbackDetail(
                effect=effect, choices=list(CHOICES),
                related_entry_id=running.id if running else None,
                reason=request.reason or None),
            state=OpenCallbackState())
        self._open[callback_id] = _Open(request, callback_id, child_id=child_id)
        projector.open_callback(entry)
        projector.set_status({
            "type": "blocked", "activeTurnId": ctx.turns.active_id or "",
            "callbackId": callback_id, "reason": "approval"})
        ctx.rpc.request("callback/call", CallbackCallParams(callback=entry),
                        lambda result, error: self._delivered(
                            callback_id, result, error))

    def _delivered(self, callback_id: str, result, error) -> None:
        """The client's answer to `callback/call`: it took the question, or
        it could not. A client that cannot show it is a no."""
        accepted = False
        if error is None and result is not None:
            try:
                accepted = CallbackCallResponse.model_validate(
                    result, by_alias=True, by_name=False).accepted
            except Exception:
                accepted = False
        if not accepted:
            self.cancel(callback_id, "The client could not show this request")

    # ---- the person answers ----------------------------------------------------

    def answer(self, params: CallbackResultParams) -> None:
        ctx = self._ctx
        callback_id = params.callback_id
        if callback_id in self._closed:
            raise RpcError(ProtocolErrorCode.CALLBACK_CLOSED,
                           f"Callback is closed: {callback_id}")
        record = self._open.get(callback_id)
        if record is None:
            raise RpcError(ProtocolErrorCode.NOT_FOUND,
                           f"Callback not found: {callback_id}")
        if params.result.error is not None:
            # The client could not answer: that is a no.
            output = {"type": "approval", "decision": {"type": "deny"},
                      "feedback": params.result.error.message}
        else:
            output = params.result.output
        try:
            parsed = _OUTPUT.validate_python(output, by_alias=True,
                                             by_name=False)
        except Exception:
            raise RpcError(ProtocolErrorCode.INVALID_PARAMS,
                           "That is not an answer to an approval.")
        if not isinstance(parsed, ApprovalCallbackOutput):
            raise RpcError(ProtocolErrorCode.CONFLICT,
                           "Callback answer has the wrong type")
        dumped = parsed.model_dump(mode="json", by_alias=True)
        if record.answer is not None:
            if record.answer == dumped:
                return
            raise RpcError(ProtocolErrorCode.CONFLICT,
                           "Callback already has a different answer")
        decision = _DECISIONS[parsed.decision.type]
        if not ctx.agent.broker.resolve(record.request.id, decision):
            # The turn ended (or the wait timed out) before the answer came.
            self._open.pop(callback_id, None)
            self._closed.add(callback_id)
            raise RpcError(ProtocolErrorCode.CALLBACK_CLOSED,
                           f"Callback is closed: {callback_id}")
        record.answer = dumped
        projector = ctx.projector
        projector.close_callback(callback_id, {"status": "answered",
                                               "output": dumped})
        self._open.pop(callback_id, None)
        if record.child_id and record.child_id in projector.child_sessions:
            cs = projector.child_sessions[record.child_id]
            if isinstance(cs.status, BlockedSessionStatus):
                turn_id = cs.status.active_turn_id or ctx.turns.active_id or ""
                unblocked_cs = cs.model_copy(update={
                    "status": RunningSessionStatus(active_turn_id=turn_id),
                    "updated_at": now_ms()
                })
                projector.child_sessions[record.child_id] = unblocked_cs
                projector._emit("session/childSessionUpdated", ChildSessionUpdatedParams,
                                child_session=unblocked_cs)
        projector.set_status({"type": "running",
                              "activeTurnId": ctx.turns.active_id or ""})
        if parsed.decision.type is ApprovalDecisionType.CANCEL_TURN:
            ctx.turns.cancel_active()

    # ---- the question went away ------------------------------------------------

    def cancel(self, callback_id: str, reason: str) -> None:
        record = self._open.pop(callback_id, None)
        if record is None:
            return
        self._closed.add(callback_id)
        self._ctx.agent.broker.resolve(record.request.id, REFUSE)
        projector = self._ctx.projector
        if record.child_id and projector and record.child_id in projector.child_sessions:
            cs = projector.child_sessions[record.child_id]
            if isinstance(cs.status, BlockedSessionStatus):
                turn_id = cs.status.active_turn_id or self._ctx.turns.active_id or ""
                unblocked_cs = cs.model_copy(update={
                    "status": RunningSessionStatus(active_turn_id=turn_id),
                    "updated_at": now_ms()
                })
                projector.child_sessions[record.child_id] = unblocked_cs
                projector._emit("session/childSessionUpdated", ChildSessionUpdatedParams,
                                child_session=unblocked_cs)
        projector.close_callback(callback_id, {"status": "cancelled",
                                               "reason": reason})

    def cancel_all(self, reason: str) -> None:
        for callback_id in list(self._open):
            self.cancel(callback_id, reason)

    def forget(self) -> None:
        """A new conversation: nothing open, nothing remembered."""
        self._open.clear()
        self._closed.clear()
