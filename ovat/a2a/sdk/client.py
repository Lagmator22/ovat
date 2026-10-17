# ovat/a2a/sdk/client.py
"""Another agent, reached over A2A, with a synchronous face.

Same shape as tools/mcp_client.MCPStdioServer: one daemon thread runs one
event loop, and every call goes in with run_coroutine_threadsafe. That
matters because a tool function is called from different places: the
native loop's thread, a LangGraph worker thread, and the Agents SDK's own
running event loop, where a plain asyncio.run() would fail. A call into a
DIFFERENT loop on a different thread works from all of them.

Everything the SDK raises leaves this file as protocol.PeerError, and every
reply as protocol.RemoteReply, so nothing else learns the SDK's types.
"""
import asyncio
import atexit
import concurrent.futures
import threading
import uuid

import httpx
from a2a.client import (A2ACardResolver, A2AClientError,
                        A2AClientTimeoutError, AgentCardResolutionError,
                        ClientConfig, create_client)
from a2a.helpers import get_artifact_text, get_message_text
from a2a.types import (CancelTaskRequest, Message, Part, Role,
                       SendMessageRequest, TaskState as SDKState)
from google.protobuf.json_format import MessageToDict

from ovat.a2a.protocol import (PeerError, RemoteCard, RemoteReply,
                               RemoteSkill, TaskState)

# The SDK's enum names to OVAT's. Anything unknown is UNKNOWN, never a crash.
_STATES = {
    SDKState.TASK_STATE_SUBMITTED: TaskState.SUBMITTED,
    SDKState.TASK_STATE_WORKING: TaskState.WORKING,
    SDKState.TASK_STATE_INPUT_REQUIRED: TaskState.INPUT_REQUIRED,
    SDKState.TASK_STATE_AUTH_REQUIRED: TaskState.AUTH_REQUIRED,
    SDKState.TASK_STATE_COMPLETED: TaskState.COMPLETED,
    SDKState.TASK_STATE_FAILED: TaskState.FAILED,
    SDKState.TASK_STATE_CANCELED: TaskState.CANCELED,
    SDKState.TASK_STATE_REJECTED: TaskState.REJECTED,
}

CARD_TIMEOUT_S = 5.0


def _state(value) -> TaskState:
    return _STATES.get(value, TaskState.UNKNOWN)


# httpx's own text for a 401 carries a docs link on developer.mozilla.org,
# which is noise for a person who only needs to fix the token.
REFUSED_TOKEN = "the agent refused the token"


def _peer_error(exc: BaseException) -> PeerError:
    """One of four kinds, whatever the SDK or httpx raised."""
    text = str(exc)
    if isinstance(exc, PeerError):
        return exc
    if isinstance(exc, (A2AClientTimeoutError, httpx.TimeoutException,
                        concurrent.futures.TimeoutError, TimeoutError)):
        return PeerError("timeout", text)
    if isinstance(exc, AgentCardResolutionError):
        if getattr(exc, "status_code", None) in (401, 403):
            return PeerError("auth", REFUSED_TOKEN)
        return PeerError("unreachable", text)
    if isinstance(exc, httpx.TransportError):
        return PeerError("unreachable", text)
    if isinstance(exc, A2AClientError):
        if "401" in text or "403" in text:
            return PeerError("auth", REFUSED_TOKEN)
        if "Network communication error" in text:
            return PeerError("unreachable", text)
    return PeerError("protocol", text)


def _remote_card(card, url: str) -> RemoteCard:
    return RemoteCard(
        name=card.name, description=card.description, url=url,
        version=card.version, streaming=bool(card.capabilities.streaming),
        skills=tuple(RemoteSkill(s.name, s.description) for s in card.skills),
        protocol_versions=tuple(dict.fromkeys(
            i.protocol_version for i in card.supported_interfaces)))


class SDKPeer:
    """One remote agent. Remembers its conversation id between messages."""

    def __init__(self, url: str, token: str | None = None,
                 timeout: float = 900.0):
        self.url = url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._client = None
        self._http = None
        self._card = None
        self.context_id: str | None = None
        self._closed = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever,
                                        name="ovat-a2a-peer", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    # ---- the synchronous face -------------------------------------------

    def card(self, timeout: float = CARD_TIMEOUT_S) -> RemoteCard:
        """The remote Agent Card. Raises PeerError."""
        self._ensure_open()
        return self._call(self._fetch_card(), timeout)

    def send(self, text: str, metadata: dict | None = None,
             on_status=None) -> RemoteReply:
        """Send one message and wait for the final state. Raises PeerError.

        on_status(text) is called for each progress line the remote sends,
        on this object's own thread.
        """
        self._ensure_open()
        return self._call(self._send(text, metadata or {}, on_status),
                          self._timeout)

    def reset(self) -> None:
        """Forget the conversation: the next message starts a new one."""
        self.context_id = None

    def close(self) -> None:
        """Close the connection and stop the thread. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        try:
            asyncio.run_coroutine_threadsafe(
                self._aclose(), self._loop).result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    # ---- inside the loop ------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise PeerError("protocol", "this connection was closed")

    def _call(self, coro, timeout: float):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except BaseException as exc:
            future.cancel()
            raise _peer_error(exc) from exc

    async def _connect(self):
        if self._client is not None:
            return self._client
        headers = ({"Authorization": f"Bearer {self._token}"}
                   if self._token else {})
        self._http = httpx.AsyncClient(
            headers=headers, timeout=httpx.Timeout(self._timeout, connect=10))
        sdk_card = await A2ACardResolver(self._http, self.url).get_agent_card()
        self._card = _remote_card(sdk_card, self.url)
        self._client = await create_client(sdk_card, ClientConfig(
            streaming=self._card.streaming, httpx_client=self._http,
            supported_protocol_bindings=["JSONRPC", "HTTP+JSON"]))
        return self._client

    async def _fetch_card(self) -> RemoteCard:
        await self._connect()
        return self._card

    async def _send(self, text, metadata, on_status) -> RemoteReply:
        client = await self._connect()
        message = Message(role=Role.ROLE_USER, message_id=uuid.uuid4().hex,
                          parts=[Part(text=text)],
                          context_id=self.context_id or "")
        request = SendMessageRequest(message=message, metadata=metadata)
        state, status_text, answer = TaskState.UNKNOWN, "", []
        task_id = context_id = None
        extra: dict = {}
        try:
            async for response in client.send_message(request):
                kind = response.WhichOneof("payload")
                if kind == "task":
                    task = response.task
                    task_id, context_id = task.id, task.context_id
                    state = _state(task.status.state)
                    status_text = get_message_text(task.status.message)
                    answer = [get_artifact_text(a) for a in task.artifacts]
                elif kind == "status_update":
                    update = response.status_update
                    task_id = update.task_id or task_id
                    context_id = update.context_id or context_id
                    state = _state(update.status.state)
                    status_text = get_message_text(update.status.message)
                    ovat = MessageToDict(update.metadata).get("ovat")
                    if isinstance(ovat, dict):
                        extra = ovat
                    if state is TaskState.WORKING and status_text and on_status:
                        on_status(status_text)
                elif kind == "artifact_update":
                    answer.append(get_artifact_text(
                        response.artifact_update.artifact))
                elif kind == "message":
                    state = TaskState.COMPLETED
                    answer = [get_message_text(response.message)]
                    context_id = response.message.context_id or context_id
        except asyncio.CancelledError:
            if task_id:
                await self._cancel_quietly(task_id)
            raise
        if context_id:
            self.context_id = context_id
        text_out = "\n".join(a for a in answer if a)
        if state is not TaskState.COMPLETED:
            text_out = status_text or text_out
        return RemoteReply(state=state, text=text_out, task_id=task_id,
                           context_id=context_id, metadata=extra)

    async def _cancel_quietly(self, task_id: str) -> None:
        try:
            await asyncio.wait_for(
                self._client.cancel_task(CancelTaskRequest(id=task_id)), 5)
        except BaseException:
            pass

    async def _aclose(self) -> None:
        try:
            if self._client is not None:
                await self._client.close()
        finally:
            if self._http is not None:
                await self._http.aclose()
