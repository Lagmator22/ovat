# ovat/a2a/sdk/server.py
"""The A2A server: the SDK's request handling around OVAT's service.

What lives here and nowhere else:
- translating SDK objects to service.Task and back (TaskUpdater);
- the ASGI app: the card route, the JSON-RPC route, and in front of them
  the host check, the body-size cap and the bearer-token check.

Facts this relies on, each seen by running a2a-sdk 1.2.2 (see the A2A
design notes): an exception that escapes execute() reaches the caller as
an InternalError carrying the raw exception text, so service.run never
lets one out; TaskUpdater raises once a task is terminal, so every update
here tolerates that; and the SDK scopes tasks by the Starlette user name,
so one user per token keeps callers apart.
"""
import contextlib
import hmac
import json
import logging

from a2a.helpers import new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCard, Task as SDKTask, TaskState, TaskStatus
from a2a.utils import AGENT_CARD_WELL_KNOWN_PATH
from google.protobuf.json_format import ParseDict
from starlette.applications import Starlette
from starlette.authentication import AuthCredentials, SimpleUser
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from ovat.a2a.service import A2AService, Task

log = logging.getLogger(__name__)

# Older callers speak A2A 0.3; the SDK translates for them.
_COMPAT_VERSION = "0.3"
# Larger than any message admission would accept, small enough that a
# caller cannot make the server buffer megabytes before being refused.
MAX_BODY_BYTES = 256 * 1024
STATUS_PATH = "/_ovat/status"


def to_agent_card(card: dict) -> AgentCard:
    """card.py's plain JSON as the SDK's type. Unknown keys raise."""
    return ParseDict(card, AgentCard())


def _with_compat_interface(card: dict) -> dict:
    """The card plus a 0.3 entry, since the SDK serves 0.3 too."""
    out = dict(card)
    first = card["supportedInterfaces"][0]
    out["supportedInterfaces"] = [
        *card["supportedInterfaces"],
        {**first, "protocolVersion": _COMPAT_VERSION}]
    return out


class _UpdaterReporter:
    """service.Reporter on top of the SDK's TaskUpdater."""

    def __init__(self, updater: TaskUpdater):
        self.up = updater
        self._sources: list[str] = []

    def _message(self, text: str):
        return self.up.new_agent_message([new_text_part(text)])

    async def _status(self, state, text=None, ovat=None) -> None:
        try:
            await self.up.update_status(
                state, message=self._message(text) if text else None,
                metadata={"ovat": ovat} if ovat else None)
        except RuntimeError:
            # "already in a terminal state": cancelled while this update
            # was on its way. Nothing to tell anyone.
            pass

    async def accepted(self) -> None:
        await self._status(TaskState.TASK_STATE_WORKING)

    async def working(self, text: str, event: dict | None = None) -> None:
        await self._status(TaskState.TASK_STATE_WORKING, text, event)

    async def answer(self, text: str, sources: list[str]) -> None:
        self._sources = list(sources)
        try:
            await self.up.add_artifact([new_text_part(text)], name="answer",
                                       last_chunk=True)
        except RuntimeError:
            pass

    async def completed(self, trace: dict) -> None:
        await self._status(TaskState.TASK_STATE_COMPLETED, None,
                           {"trace": trace, "sources": self._sources})

    async def failed(self, text: str, trace: dict | None = None) -> None:
        await self._status(TaskState.TASK_STATE_FAILED, text,
                           {"trace": trace} if trace else None)

    async def rejected(self, text: str) -> None:
        await self._status(TaskState.TASK_STATE_REJECTED, text)


def _hops_and_chain(metadata: dict) -> tuple[int, list[str]]:
    """The caller's hop count and chain, as sent. Untrusted input: numbers
    arrive as floats (protobuf Struct), and anything odd counts as none."""
    ovat = metadata.get("ovat") if isinstance(metadata, dict) else None
    if not isinstance(ovat, dict):
        return 0, []
    try:
        hops = max(0, int(ovat.get("hops", 0)))
    except (TypeError, ValueError):
        hops = 0
    chain = ovat.get("chain")
    chain = [str(x) for x in chain[:32]] if isinstance(chain, list) else []
    return hops, chain


class OVATExecutor(AgentExecutor):
    """The SDK's hook: one call per incoming message."""

    def __init__(self, service: A2AService):
        self.service = service

    async def execute(self, context: RequestContext, event_queue) -> None:
        if context.current_task is None:
            await event_queue.enqueue_event(SDKTask(
                id=context.task_id, context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[context.message]))
        user = getattr(context.call_context, "user", None)
        caller = getattr(user, "user_name", "") or "unknown"
        hops, chain = _hops_and_chain(context.metadata)
        task = Task(task_id=context.task_id, context_id=context.context_id,
                    caller=caller, text=context.get_user_input(),
                    hops=hops, chain=chain)
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await self.service.run(task, _UpdaterReporter(updater))

    async def cancel(self, context: RequestContext, event_queue) -> None:
        self.service.cancel(context.task_id)
        try:
            await TaskUpdater(event_queue, context.task_id,
                              context.context_id).cancel()
        except RuntimeError:
            pass                               # it had already finished


def _reply(status: int, message: str, **headers) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status,
                        headers=headers or None)


class _Guard:
    """Bearer token on everything but the card, and a body-size cap.

    Plain ASGI rather than Starlette's AuthenticationMiddleware so a missing
    token is a JSON 401 the calling agent can read, and so the size check
    happens before the SDK reads the body.
    """

    def __init__(self, app, callers: dict[str, str],
                 max_body: int = MAX_BODY_BYTES):
        self.app = app
        self._callers = [(t.encode(), name) for t, name in callers.items()]
        self._max_body = max_body

    def _caller(self, header: bytes) -> str | None:
        if not header.startswith(b"Bearer "):
            return None
        sent = header[len(b"Bearer "):].strip()
        found = None
        for token, name in self._callers:     # no early exit: timing-safe
            if hmac.compare_digest(sent, token):
                found = name
        return found

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        public = (scope["method"] == "GET"
                  and scope["path"] == AGENT_CARD_WELL_KNOWN_PATH)
        if not public:
            caller = self._caller(headers.get(b"authorization", b""))
            if caller is None:
                response = _reply(401, "missing or wrong bearer token",
                                  **{"WWW-Authenticate": "Bearer"})
                return await response(scope, receive, send)
            scope["user"] = SimpleUser(caller)
            scope["auth"] = AuthCredentials(["authenticated"])
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = 0
        if length > self._max_body:
            response = _reply(413, f"request larger than {self._max_body} "
                                   f"bytes")
            return await response(scope, receive, send)
        return await self.app(scope, _capped(receive, self._max_body), send)


def _capped(receive, limit: int):
    """`receive`, but a body that streams past `limit` is cut off as a
    disconnect (no Content-Length does not mean no limit)."""
    seen = 0

    async def wrapped():
        nonlocal seen
        message = await receive()
        if message["type"] == "http.request":
            seen += len(message.get("body", b""))
            if seen > limit:
                return {"type": "http.disconnect"}
        return message
    return wrapped


def build_app(service: A2AService, card: dict, callers: dict[str, str],
              allowed_hosts: list[str]):
    """The whole server as one ASGI app, ready for uvicorn."""
    sdk_card = to_agent_card(_with_compat_interface(card))
    handler = DefaultRequestHandler(agent_executor=OVATExecutor(service),
                                    task_store=InMemoryTaskStore(),
                                    agent_card=sdk_card)

    async def status(request):
        return JSONResponse(service.status())

    routes = [*create_agent_card_routes(sdk_card),
              Route(STATUS_PATH, status, methods=["GET"]),
              *create_jsonrpc_routes(handler, "/", enable_v0_3_compat=True)]

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        try:
            await handler.aclose()
        finally:
            service.close()

    app = Starlette(routes=routes, lifespan=lifespan)
    # Order matters: the host check runs first, so a page in the user's
    # browser that rebinds a name to 127.0.0.1 is refused before anything
    # else looks at it.
    return TrustedHostMiddleware(_Guard(app, callers),
                                 allowed_hosts=allowed_hosts)


def card_json(card: dict) -> str:
    """The card exactly as served, for `ovat a2a card`."""
    from google.protobuf.json_format import MessageToDict
    return json.dumps(MessageToDict(to_agent_card(
        _with_compat_interface(card))), indent=2)
