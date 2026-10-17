# ovat/tui/server/rpc.py
"""The JSON-RPC loop of OVAT's app server.

Note to myself: the copied TUI is a CLIENT. It sends JSON-RPC requests over a
transport and reads notifications back. This file is the other end of that
wire, and it knows nothing about agents: it reads a message, finds the
handler for its method, checks the params with the protocol model, and
writes the answer. The handlers live in ovat/tui/server/methods/.

What goes on the wire follows what the copied client expects (checked in
client.py, connection.py and session.py, and against Vibe's own server for
the framing, which is only read, never imported):

- a request gets exactly one response, `{"jsonrpc","id","result"}` or
  `{"jsonrpc","id","error":{"code","message","data"}}`; the code is a
  ProtocolErrorCode string ("method_not_found", ...)
- results and notification params are dumped by alias (camelCase), the way
  the client reads them
- the server can ask the client something (`callback/call`). Those requests
  carry integer ids; the client's own requests carry string ids
- every message leaves through ONE outbox in the order it was queued, so a
  response is never overtaken by the notifications queued after it

Requests run as tasks, because some are long (`session/ready/wait` waits for
the model, `session/shellCommand` for the command). `initialize` is the one
that runs inline: the `initialized` notification right behind it must see it
done.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue, ValidationError

from ovat.tui.vibe.app_server._model import ProtocolModel, validate_wire
from ovat.tui.vibe.app_server.protocol import (SERVER_METHODS,
                                               EventNotificationParams,
                                               InitializeParams,
                                               InitializeResponse,
                                               InvalidParamsData,
                                               InvalidParamsIssue,
                                               JsonRpcErrorResponse,
                                               JsonRpcSuccessResponse,
                                               Notification, ProtocolError,
                                               ProtocolErrorCode, ServerInfo,
                                               ServerRequest,
                                               validate_json_rpc_envelope)
from ovat.tui.vibe.app_server.transport import JsonRpcTransport


class RpcError(Exception):
    """A request the server refuses, with the code the client switches on and
    a sentence a person can read."""

    def __init__(self, code: ProtocolErrorCode, message: str,
                 data: JsonValue = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


def not_supported(method: str) -> RpcError:
    """The answer for every method OVAT does not serve.

    A method that exists in Vibe's protocol gets NOT_IMPLEMENTED, a name the
    protocol does not know gets METHOD_NOT_FOUND. The copied client treats
    both as "this backend does not have it" (plugins/read, for one), so the
    screens that ask degrade instead of failing."""
    code = (ProtocolErrorCode.NOT_IMPLEMENTED if method in SERVER_METHODS
            else ProtocolErrorCode.METHOD_NOT_FOUND)
    return RpcError(code, f"not supported in OVAT: {method}")


@dataclass(frozen=True)
class Reply:
    """A response, plus work to do once it has been queued.

    Vibe's server sends a response first and the events it causes after
    (a turn answers `turn/start`, then announces itself); `after` is how a
    handler asks for that order."""

    response: ProtocolModel
    after: Callable[[], None] | None = None


Handler = Callable[[Any, Any], Awaitable["ProtocolModel | Reply"]]


@dataclass(frozen=True)
class Method:
    name: str
    params: type[ProtocolModel]
    handler: Handler


class Routes:
    """method name -> (params model, handler). One table for the whole server."""

    def __init__(self) -> None:
        self._methods: dict[str, Method] = {}

    def method(self, name: str, params: type[ProtocolModel]):
        def register(handler: Handler) -> Handler:
            if name in self._methods:
                raise ValueError(f"method registered twice: {name}")
            self._methods[name] = Method(name, params, handler)
            return handler
        return register

    def merge(self, other: Routes) -> None:
        for name, method in other._methods.items():
            if name in self._methods:
                raise ValueError(f"method registered twice: {name}")
            self._methods[name] = method

    def get(self, name: str) -> Method | None:
        return self._methods.get(name)

    def names(self) -> list[str]:
        return sorted(self._methods)


_NEW, _INITIALIZE_RECEIVED, _READY = range(3)

# What the client answers to a request the server made: the result dict, or
# the error it sent back.
ClientAnswer = Callable[[dict | None, ProtocolError | None], None]


class RpcServer:
    def __init__(self, transport: JsonRpcTransport, routes: Routes, ctx: Any,
                 *, on_close: Callable[[], Awaitable[None]] | None = None
                 ) -> None:
        self._transport = transport
        self._routes = routes
        self._ctx = ctx
        self._on_close = on_close
        self._state = _NEW
        self._outbox: asyncio.Queue[dict | None] = asyncio.Queue()
        self._tasks: set[asyncio.Task] = set()
        self._pending: dict[int, ClientAnswer] = {}
        self._next_id = 1
        self._stop = asyncio.Event()
        self._finished = False

    # ---- what handlers and the projector use -----------------------------------

    def notify(self, method: str, params: ProtocolModel) -> None:
        """Queue a notification. Never blocks; no-op once the server closed.
        A client can switch off a kind of notification at the handshake
        (`disabledNotifications`); events that keep its numbering straight
        are never switched off."""
        if (method in self._ctx.capabilities.disabled_notifications
                and not isinstance(params, EventNotificationParams)):
            return
        self._send({"jsonrpc": "2.0", "method": method,
                    "params": params.model_dump(mode="json", by_alias=True)})

    def request(self, method: str, params: ProtocolModel,
                answer: ClientAnswer) -> int:
        """Ask the client something (a callback). `answer` gets the result
        dict, or the error the client sent."""
        request_id = self._next_id
        self._next_id += 1
        self._pending[request_id] = answer
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method,
                    "params": params.model_dump(mode="json", by_alias=True)})
        return request_id

    def stop(self) -> None:
        """End the loop once everything queued has been written."""
        self._stop.set()

    @property
    def initialized(self) -> bool:
        return self._state == _READY

    @property
    def stopping(self) -> bool:
        """True once the server has been asked to end (or is ending)."""
        return self._stop.is_set() or self._finished

    # ---- the loop --------------------------------------------------------------

    async def serve(self) -> None:
        writer = asyncio.create_task(self._write_loop(), name="ovat-rpc-writer")
        reader = asyncio.create_task(self._read_loop(), name="ovat-rpc-reader")
        stopper = asyncio.create_task(self._stop.wait(), name="ovat-rpc-stop")
        try:
            await asyncio.wait({reader, stopper},
                               return_when=asyncio.FIRST_COMPLETED)
        finally:
            await self._shutdown(reader, stopper, writer)

    async def _shutdown(self, reader: asyncio.Task, stopper: asyncio.Task,
                        writer: asyncio.Task) -> None:
        self._finished = True
        for task in (reader, stopper, *self._tasks):
            task.cancel()
        await asyncio.gather(reader, stopper, *self._tasks,
                             return_exceptions=True)
        self._pending.clear()
        try:
            if self._on_close is not None:
                await self._on_close()
        finally:
            # Whatever was queued before the close (the response to
            # session/stop, say) still goes out; then the writer ends.
            self._outbox.put_nowait(None)
            await asyncio.gather(writer, return_exceptions=True)
            try:
                await self._transport.close()
            except Exception:
                pass

    async def _read_loop(self) -> None:
        async for raw in self._transport.messages():
            try:
                message = validate_json_rpc_envelope(raw)
            except ValidationError:
                self._reject_unreadable(raw)
                continue
            if isinstance(message, ServerRequest):
                if message.method == "initialize":
                    await self._handle_request(message)
                else:
                    task = asyncio.create_task(self._handle_request(message))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
            elif isinstance(message, Notification):
                self._handle_notification(message)
            else:
                self._handle_response(message)

    def _reject_unreadable(self, raw: object) -> None:
        request_id = raw.get("id") if isinstance(raw, dict) else None
        if isinstance(request_id, (int, str)) and not isinstance(request_id,
                                                                  bool):
            self._send_error(request_id, RpcError(
                ProtocolErrorCode.INVALID_REQUEST,
                "That is not a JSON-RPC 2.0 message."))

    def _handle_notification(self, notification: Notification) -> None:
        # Only `initialized` means anything. A JSON-RPC notification has no
        # answer, so anything else is ignored rather than punished.
        if (notification.method == "initialized"
                and self._state == _INITIALIZE_RECEIVED):
            self._state = _READY

    def _handle_response(self, response: Any) -> None:
        if not isinstance(response, (JsonRpcSuccessResponse,
                                     JsonRpcErrorResponse)):
            return
        answer = (self._pending.pop(response.id, None)
                  if isinstance(response.id, int) else None)
        if answer is None:
            return
        try:
            if isinstance(response, JsonRpcErrorResponse):
                answer(None, response.error)
            else:
                answer(dict(response.result), None)
        except Exception as exc:
            # One bad answer must not end the loop that reads the rest.
            self._ctx.report_error(f"{type(exc).__name__}: {exc}")

    # ---- one request -----------------------------------------------------------

    async def _handle_request(self, request: ServerRequest) -> None:
        try:
            reply = await self._call(request)
            response, after = ((reply.response, reply.after)
                               if isinstance(reply, Reply) else (reply, None))
            result = response.model_dump(mode="json", by_alias=True)
        except asyncio.CancelledError:
            raise
        except RpcError as exc:
            self._send_error(request.id, exc)
            return
        except Exception as exc:
            # A bug in a handler: the person gets the sentence, never a
            # traceback (rule 7).
            self._send_error(request.id, RpcError(
                ProtocolErrorCode.INTERNAL_ERROR,
                f"{type(exc).__name__}: {exc}"))
            return
        self._send({"jsonrpc": "2.0", "id": request.id, "result": result})
        if after is not None:
            try:
                after()
            except Exception as exc:
                self._ctx.report_error(
                    f"{request.method}: {type(exc).__name__}: {exc}")

    async def _call(self, request: ServerRequest) -> ProtocolModel | Reply:
        if request.method == "initialize":
            return self._initialize(request.params)
        if self._state != _READY:
            raise RpcError(ProtocolErrorCode.NOT_INITIALIZED,
                           "initialize must be the first request")
        method = self._routes.get(request.method)
        if method is None:
            raise not_supported(request.method)
        try:
            params = validate_wire(method.params, request.params)
        except ValidationError as exc:
            raise _invalid_params(exc) from None
        return await method.handler(self._ctx, params)

    def _initialize(self, raw_params: dict) -> InitializeResponse:
        if self._state != _NEW:
            raise RpcError(ProtocolErrorCode.INVALID_REQUEST,
                           "initialize may only be called once")
        try:
            params = validate_wire(InitializeParams, raw_params)
        except ValidationError as exc:
            raise _invalid_params(exc) from None
        self._ctx.on_initialize(params)
        self._state = _INITIALIZE_RECEIVED
        return InitializeResponse(
            server_info=ServerInfo(name="ovat-app-server", version=_version()),
            process_id=os.getpid())

    # ---- the outbox ------------------------------------------------------------

    def _send_error(self, request_id: int | str, error: RpcError) -> None:
        body = ProtocolError(code=error.code, message=str(error),
                             data=error.data)
        self._send({"jsonrpc": "2.0", "id": request_id,
                    "error": body.model_dump(mode="json", by_alias=True)})

    def _send(self, message: dict) -> None:
        if not self._finished:
            self._outbox.put_nowait(message)

    async def _write_loop(self) -> None:
        broken = False
        while True:
            message = await self._outbox.get()
            if message is None:
                return
            if broken:
                continue
            try:
                await self._transport.send(message)
            except Exception:
                # The client went away. Keep draining so nothing waits on us.
                broken = True


def _invalid_params(exc: ValidationError) -> RpcError:
    """A request whose params the protocol model refused, with the path of
    every field that failed (the client prints them). Only the request's own
    params are reported this way: a model the SERVER fails to build is a bug
    here, not the caller's mistake."""
    issues = [InvalidParamsIssue(
        path=[p if isinstance(p, (int, str)) else str(p) for p in issue["loc"]],
        message=issue["msg"]) for issue in exc.errors()]
    return RpcError(
        ProtocolErrorCode.INVALID_PARAMS, "Invalid request parameters",
        InvalidParamsData(error_count=exc.error_count(), issues=issues)
        .model_dump(mode="json", by_alias=True))


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("ovat")
    except PackageNotFoundError:
        return "unknown"
