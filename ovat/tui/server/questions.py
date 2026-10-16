# ovat/tui/server/questions.py
"""Questions: the ask_user tool's QuestionBroker, shown as Vibe's question
callback.

Note to myself: the same dance as approvals.py. The agent's worker thread
BLOCKS inside QuestionBroker.__call__ until somebody calls
broker.resolve(id, answer). Vibe calls the moment a "user_input" callback:
the server sends `callback/call` with a callback entry whose detail is a
UserQuestionRequest, the person answers in the copied question widget, the
client sends `callback/result` with a UserQuestionResult, and that resolves
the broker here.

    answered           -> the answer text
    cancelled (Esc)    -> None ("no answer"; the tool tells the model)
    client cannot ask  -> None, and a warning in the transcript

The question itself is built by bridge.question_request (Vibe wants at
least two options and pads the rest with "Not sure" / "Skip").
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter

from ovat.agent.questions import QuestionRequest
from ovat.tui.server.bridge import question_request
from ovat.tui.server.projector import now_ms
from ovat.tui.server.rpc import RpcError
from ovat.tui.vibe.app_server.models import (CallbackOutput,
                                             OpenCallbackState,
                                             PublicCallbackEntry,
                                             PublicEntryGenerationStatus,
                                             UserInputCallbackDetail,
                                             UserInputCallbackOutput)
from ovat.tui.vibe.app_server.protocol import (CallbackCallParams,
                                               CallbackCallResponse,
                                               CallbackResultParams,
                                               ProtocolErrorCode)

PREFIX = "question-"
_OUTPUT = TypeAdapter(CallbackOutput)


@dataclass
class _Open:
    request: QuestionRequest
    callback_id: str
    answer: dict | None = None


class Questions:
    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self._open: dict[str, _Open] = {}
        self._closed: set[str] = set()

    @staticmethod
    def owns(callback_id: str) -> bool:
        return callback_id.startswith(PREFIX)

    # ---- the broker asks -------------------------------------------------------

    def request(self, request: QuestionRequest) -> None:
        ctx = self._ctx
        projector = ctx.projector
        broker = ctx.agent.questions
        if projector is None or ctx.rpc is None:
            broker.resolve(request.id, None)
            return
        if "user_input" not in ctx.capabilities.callback_kinds:
            broker.resolve(request.id, None)
            projector.warning(
                "The agent wanted to ask you a question, and this client "
                "cannot show one. It was told nobody answered.")
            return
        callback_id = f"{PREFIX}{request.id}"
        running = projector.bridge.running("ask_user")
        asked = question_request(request.question, request.choices)
        stamp = now_ms()
        entry = PublicCallbackEntry(
            id=f"callback:{callback_id}", session_id=projector.session.id,
            turn_id=ctx.turns.active_id, created_at=stamp, updated_at=stamp,
            generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
            related_entry_id=running.id if running else None,
            callback_id=callback_id, title="The agent has a question",
            detail=UserInputCallbackDetail(
                request=asked,
                related_entry_id=running.id if running else None),
            state=OpenCallbackState())
        self._open[callback_id] = _Open(request, callback_id)
        projector.open_callback(entry)
        projector.set_status({
            "type": "blocked", "activeTurnId": ctx.turns.active_id or "",
            "callbackId": callback_id, "reason": "user_input"})
        ctx.rpc.request("callback/call", CallbackCallParams(callback=entry),
                        lambda result, error: self._delivered(
                            callback_id, result, error))

    def _delivered(self, callback_id: str, result, error) -> None:
        accepted = False
        if error is None and result is not None:
            try:
                accepted = CallbackCallResponse.model_validate(
                    result, by_alias=True, by_name=False).accepted
            except Exception:
                accepted = False
        if not accepted:
            self.cancel(callback_id, "The client could not show this question")

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
            output = {"type": "user_input",
                      "result": {"answers": [], "cancelled": True}}
        else:
            output = params.result.output
        try:
            parsed = _OUTPUT.validate_python(output, by_alias=True,
                                             by_name=False)
        except Exception:
            raise RpcError(ProtocolErrorCode.INVALID_PARAMS,
                           "That is not an answer to a question.")
        if not isinstance(parsed, UserInputCallbackOutput):
            raise RpcError(ProtocolErrorCode.CONFLICT,
                           "Callback answer has the wrong type")
        dumped = parsed.model_dump(mode="json", by_alias=True)
        if record.answer is not None:
            if record.answer == dumped:
                return
            raise RpcError(ProtocolErrorCode.CONFLICT,
                           "Callback already has a different answer")
        result = parsed.result
        text = None
        if not result.cancelled and result.answers:
            text = "; ".join(a.answer for a in result.answers)
        if not ctx.agent.questions.resolve(record.request.id, text):
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
        projector.set_status({"type": "running",
                              "activeTurnId": ctx.turns.active_id or ""})

    # ---- the question went away ------------------------------------------------

    def cancel(self, callback_id: str, reason: str) -> None:
        record = self._open.pop(callback_id, None)
        if record is None:
            return
        self._closed.add(callback_id)
        self._ctx.agent.questions.resolve(record.request.id, None)
        self._ctx.projector.close_callback(callback_id, {
            "status": "cancelled", "reason": reason})

    def cancel_all(self, reason: str) -> None:
        for callback_id in list(self._open):
            self.cancel(callback_id, reason)

    def forget(self) -> None:
        self._open.clear()
        self._closed.clear()
