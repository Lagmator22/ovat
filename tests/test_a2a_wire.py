# tests/test_a2a_wire.py
"""A2A over a real socket: OVAT's server, the SDK client, OVAT's tool.

One uvicorn server on a loopback port per module, serving a native agent on
a scripted model (no OVMS). Everything the network adds is checked here:
the card is public, every other path needs a token, a wrong Host header or
an oversized body is refused before the SDK sees it, and a `type: a2a`
tool in another workflow really reaches the server and back.
"""
import socket
import threading
import time

import pytest

pytest.importorskip("a2a")
uvicorn = pytest.importorskip("uvicorn")
httpx = pytest.importorskip("httpx")

from ovat.a2a.card import card_dict  # noqa: E402
from ovat.a2a.protocol import PeerError, TaskState  # noqa: E402
from ovat.a2a.sdk.client import SDKPeer  # noqa: E402
from ovat.a2a.sdk.server import build_app  # noqa: E402
from ovat.a2a.service import A2AService  # noqa: E402
from ovat.agent.events import EventBus  # noqa: E402
from ovat.agent.factory import build_agent, close_agent, observe_tools  # noqa: E402
from ovat.agent.loop import AgentLoop  # noqa: E402
from ovat.config.workflow import A2AServerConfig, WorkflowConfig  # noqa: E402
from tests.conftest import FakeLLMProvider, make_tool_call, reply  # noqa: E402

TOKEN_A, TOKEN_B = "token-for-planner", "token-for-other"


class AllowAll:
    def check(self, ask):
        return None


class Refuse:
    def check(self, ask):
        return "message longer than 8000 characters (a2a.server.max_message_chars)"


class GatedLLM(FakeLLMProvider):
    """Scripted: one search_docs call, then an answer naming the question."""

    def __init__(self):
        super().__init__([])

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        if messages[-1]["role"] == "user":
            return reply("tool_calls", tool_calls=[
                make_tool_call("c1", "search_docs", {"query": "leave"})])
        return reply("stop", content=f"answer to: {messages[1]['content']}")


def _agent():
    bus = EventBus()
    tools = observe_tools({"search_docs": {
        "schema": {"type": "function", "function": {
            "name": "search_docs", "description": "Search the handbook.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}}, "required": ["query"]}}},
        "function": lambda query: [{"text": "20 days",
                                    "source": "/home/u/hr/leave.md",
                                    "distance": 0.1}],
        "kind": "builtin"}}, bus)
    agent = AgentLoop(GatedLLM(), tools, system_prompt="SECRET PROMPT",
                      stop=bus.stop)
    agent.events = bus
    agent.mcp_servers = []
    return agent


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(admission):
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    settings = A2AServerConfig(port=port, audit_log=None)
    service = A2AService(_agent, settings, admission=admission)
    cfg = WorkflowConfig(model={"name": "m"},
                         agent={"system_prompt": "SECRET PROMPT"})
    card = card_dict(cfg, settings, _agent().tools, url, "team-docs", "1.2.0")
    app = build_app(service, card, {TOKEN_A: "planner", TOKEN_B: "other"},
                    ["127.0.0.1", "localhost"])
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    return server, thread, url


@pytest.fixture(scope="module")
def server_url():
    server, thread, url = _serve(AllowAll())
    yield url
    server.should_exit = True
    thread.join(10)


@pytest.fixture(scope="module")
def refusing_url():
    server, thread, url = _serve(Refuse())
    yield url
    server.should_exit = True
    thread.join(10)


# ---- the HTTP surface ------------------------------------------------------

def test_the_card_is_public_and_holds_no_secret(server_url):
    r = httpx.get(server_url + ".well-known/agent-card.json")
    assert r.status_code == 200
    card = r.json()
    assert card["name"] == "team-docs"
    assert "SECRET" not in r.text
    versions = {i["protocolVersion"] for i in card["supportedInterfaces"]}
    assert versions == {"1.0", "0.3"}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"},
                                     {"Authorization": TOKEN_A}])
def test_everything_else_needs_the_token(server_url, headers):
    r = httpx.post(server_url, json={"jsonrpc": "2.0", "id": 1,
                                     "method": "GetTask",
                                     "params": {"id": "x"}},
                   headers={"A2A-Version": "1.0", **headers})
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_a_foreign_host_header_is_refused(server_url):
    # A page in the user's browser that rebinds evil.example to 127.0.0.1
    # still sends Host: evil.example.
    r = httpx.get(server_url + ".well-known/agent-card.json",
                  headers={"Host": "evil.example"})
    assert r.status_code == 400


def test_an_oversized_body_is_refused_before_parsing(server_url):
    r = httpx.post(server_url, content=b"x" * (300 * 1024),
                   headers={"Authorization": f"Bearer {TOKEN_A}",
                            "Content-Type": "application/json"})
    assert r.status_code == 413


def test_the_status_route_needs_the_token(server_url):
    assert httpx.get(server_url + "_ovat/status").status_code == 401
    r = httpx.get(server_url + "_ovat/status",
                  headers={"Authorization": f"Bearer {TOKEN_A}"})
    assert r.status_code == 200 and "queued" in r.json()


# ---- the SDK client against OVAT's server ----------------------------------

def test_a_question_comes_back_with_progress_trace_and_file_names(server_url):
    peer = SDKPeer(server_url, token=TOKEN_A, timeout=30)
    lines = []
    try:
        assert peer.card().name == "team-docs"
        result = peer.send("How many leave days?", on_status=lines.append)
    finally:
        peer.close()
    assert result.state is TaskState.COMPLETED
    assert result.text == "answer to: How many leave days?"
    assert "calling search_docs" in lines
    assert result.metadata["sources"] == ["leave.md"]   # a name, not a path
    assert result.metadata["trace"]["tool_calls"] == 1


def test_a_follow_up_continues_the_same_conversation(server_url):
    peer = SDKPeer(server_url, token=TOKEN_A, timeout=30)
    try:
        first = peer.send("one")
        second = peer.send("two")
    finally:
        peer.close()
    assert first.context_id and second.context_id == first.context_id


def test_one_caller_cannot_read_anothers_task(server_url):
    peer = SDKPeer(server_url, token=TOKEN_A, timeout=30)
    try:
        task_id = peer.send("mine").task_id
    finally:
        peer.close()
    body = {"jsonrpc": "2.0", "id": 1, "method": "GetTask",
            "params": {"id": task_id}}
    own = httpx.post(server_url, json=body, headers={
        "A2A-Version": "1.0", "Authorization": f"Bearer {TOKEN_A}"}).json()
    other = httpx.post(server_url, json=body, headers={
        "A2A-Version": "1.0", "Authorization": f"Bearer {TOKEN_B}"}).json()
    assert "result" in own
    assert "error" in other


def test_a_wrong_token_is_an_auth_error(server_url):
    peer = SDKPeer(server_url, token="wrong", timeout=30)
    try:
        with pytest.raises(PeerError) as caught:
            peer.send("hi")
    finally:
        peer.close()
    assert caught.value.kind == "auth"
    assert "mozilla" not in caught.value.detail.lower()


def test_a_refused_token_is_one_plain_sentence_without_a_docs_link():
    from a2a.client import AgentCardResolutionError
    from ovat.a2a.sdk.client import _peer_error

    noisy = AgentCardResolutionError(
        "Client error '401 Unauthorized' for url 'http://x/card'\nFor more "
        "information check: https://developer.mozilla.org/en-US/docs/Web/"
        "HTTP/Status/401", status_code=401)
    error = _peer_error(noisy)
    assert error.kind == "auth"
    assert error.detail == "the agent refused the token"


def test_nothing_listening_is_unreachable():
    peer = SDKPeer(f"http://127.0.0.1:{_free_port()}", token="t", timeout=5)
    try:
        with pytest.raises(PeerError) as caught:
            peer.card()
    finally:
        peer.close()
    assert caught.value.kind == "unreachable"


def test_a_refusal_arrives_as_rejected_with_the_reason(refusing_url):
    peer = SDKPeer(refusing_url, token=TOKEN_A, timeout=30)
    try:
        result = peer.send("hi")
    finally:
        peer.close()
    assert result.state is TaskState.REJECTED
    assert "max_message_chars" in result.text


def test_an_old_0_3_caller_is_still_served(server_url):
    r = httpx.post(server_url, headers={"Authorization": f"Bearer {TOKEN_A}"},
                   json={"jsonrpc": "2.0", "id": 1, "method": "message/send",
                         "params": {"message": {
                             "kind": "message", "role": "user",
                             "messageId": "m1",
                             "parts": [{"kind": "text", "text": "old"}]}}})
    assert r.status_code == 200
    assert r.json()["result"]["status"]["state"] == "completed"


# ---- a `type: a2a` tool in another workflow, end to end -------------------

def test_a_workflow_tool_asks_the_server_and_reads_the_framed_reply(
        server_url, monkeypatch):
    monkeypatch.setenv("PLANNER_TOKEN", TOKEN_A)
    cfg = WorkflowConfig(model={"name": "m"}, tools=[{
        "name": "team_docs", "type": "a2a", "url": server_url,
        "token_env": "PLANNER_TOKEN", "timeout": 30}])
    agent = build_agent(cfg)
    try:
        schema = agent.tools["team_docs"]["schema"]["function"]
        assert 'Remote agent "team-docs"' in schema["description"]
        result = agent.tools["team_docs"]["function"](
            message="How many leave days?")
    finally:
        close_agent(agent)
    assert result.startswith('[Reply from remote agent "team_docs"')
    assert result.endswith("answer to: How many leave days?")
