# tests/test_agent_helper_tools.py
"""todo, ask_user, web_fetch, web_search and the project context.

Note to myself: the tools are plain functions, so these tests call them
directly. The web ones talk to a throwaway http.server on 127.0.0.1 or to a
faked `_request`; nothing here reaches the internet.
"""
import http.server
import threading
import urllib.error

import pytest

from ovat.a2a import exposure
from ovat.agent import factory
from ovat.agent.events import EventBus, observed
from ovat.agent.permissions import default_permission
from ovat.agent.questions import QuestionBroker
from ovat.config.workflow import WorkflowConfig
from ovat.tools import web_search as ws
from ovat.tools.ask_user import NO_PERSON, ask_user_impl
from ovat.tools.project import Shown
from ovat.tools.todo import TodoStore, todo_impl
from ovat.tools.web_fetch import html_to_text, web_fetch_impl


def _is_error(text):
    return str(text).startswith("Error:")


# ---- todo -----------------------------------------------------------------

def _item(i, status="pending", priority="medium"):
    return {"id": str(i), "content": f"step {i}", "status": status,
            "priority": priority}


def test_todo_write_replaces_and_read_returns_the_list():
    store = TodoStore()
    out = todo_impl("write", [_item(1), _item(2, "in_progress", "high")],
                    store)
    assert out.startswith("Updated 2 todos")
    assert out.display["kind"] == "todo" and len(out.display["todos"]) == 2
    todo_impl("write", [_item(3, "completed")], store)
    read = todo_impl("read", None, store)
    assert read.startswith("Retrieved 1 todos") and "step 3" in read
    assert "step 1" not in read


def test_todo_fills_defaults_and_refuses_bad_input():
    store = TodoStore()
    todo_impl("write", [{"id": "a", "content": "x"}], store)
    assert store.items[0]["status"] == "pending"
    assert store.items[0]["priority"] == "medium"
    for bad in ([_item(1), _item(1)],                       # duplicate ids
                [{"id": "a"}],                              # no content
                [_item(1, status="done")],                  # bad status
                [_item(1, priority="urgent")],              # bad priority
                ["text"],                                   # not an object
                [_item(i) for i in range(101)]):            # too many
        assert _is_error(todo_impl("write", bad, store))
    assert _is_error(todo_impl("write", None, store))
    assert _is_error(todo_impl("rename", None, store))
    assert len(store.items) == 1            # a refused write changes nothing


def test_each_built_agent_has_its_own_todo_list():
    cfg = WorkflowConfig.model_validate({
        "model": {"name": "m"}, "tools": [{"name": "todo"}]})
    one = factory.build_tools(cfg, events=EventBus())["todo"]["function"]
    two = factory.build_tools(cfg, events=EventBus())["todo"]["function"]
    one("write", todos=[_item(1)])
    assert "step 1" in one("read")
    assert str(two("read")).startswith("Retrieved 0")


# ---- ask_user -------------------------------------------------------------

def test_ask_user_with_nobody_to_ask_answers_at_once():
    bus = EventBus()                         # no asker: ovat run, bench, A2A
    assert ask_user_impl("Which one?", ["a", "b"], bus) == NO_PERSON
    assert NO_PERSON == ("Error: there is no person to ask here; decide "
                         "yourself or say what you need.")


def test_ask_user_returns_the_answer_and_marks_a_typed_one():
    bus = EventBus()
    bus.asker = lambda question, choices: "sqlite"
    out = ask_user_impl("Which db?", ["sqlite", "postgres"], bus)
    assert "sqlite" in out and out.display["cancelled"] is False
    assert out.display["is_other"] is False
    bus.asker = lambda question, choices: "duckdb"
    assert ask_user_impl("Which db?", ["sqlite", "postgres"],
                         bus).display["is_other"] is True


def test_ask_user_cancelled_or_failing_asker_is_a_sentence_not_a_crash():
    bus = EventBus()
    bus.asker = lambda q, c: None
    out = ask_user_impl("Which?", None, bus)
    assert "did not answer" in out and out.display["cancelled"] is True

    def boom(q, c):
        raise RuntimeError("ui died")
    bus.asker = boom
    assert "did not answer" in ask_user_impl("Which?", None, bus)
    assert _is_error(ask_user_impl("  ", None, bus))


def test_broker_blocks_until_resolved_and_cancel_answers_none():
    seen = []
    broker = QuestionBroker(on_request=seen.append, timeout=5)
    result = []
    worker = threading.Thread(
        target=lambda: result.append(broker("Pick?", ["a", "b"])))
    worker.start()
    deadline = threading.Event()
    while not seen and not deadline.wait(0.01):
        pass
    assert seen[0].question == "Pick?" and seen[0].choices == ["a", "b"]
    assert broker.resolve(seen[0].id, "a") is True
    worker.join(5)
    assert result == ["a"]
    assert broker.resolve(seen[0].id, "b") is False        # only once

    seen.clear()
    result.clear()
    worker = threading.Thread(target=lambda: result.append(broker("Again?", [])))
    worker.start()
    while not seen and not deadline.wait(0.01):
        pass
    broker.cancel_all()                                     # Esc
    worker.join(5)
    assert not worker.is_alive() and result == [None]


def test_broker_without_a_listener_and_a_timeout_never_hang():
    assert QuestionBroker()("anyone?", []) is None
    quiet = QuestionBroker(on_request=lambda r: None, timeout=0.05)
    assert quiet("anyone?", []) is None
    assert quiet.pending() == []


def test_ask_user_through_a_built_agent_reads_the_asker_at_call_time():
    cfg = WorkflowConfig.model_validate({
        "model": {"name": "m"}, "tools": [{"name": "ask_user"}]})
    bus = EventBus()
    tools = factory.observe_tools(factory.build_tools(cfg, events=bus), bus)
    ask = tools["ask_user"]["function"]
    assert ask(question="Which?") == NO_PERSON
    bus.asker = lambda q, c: "yes"                # the TUI connects later
    assert "yes" in ask(question="Which?")


# ---- web_fetch ------------------------------------------------------------

PAGE = (b"<html><head><title>Docs</title><style>p{}</style></head><body>"
        b"<script>evil()</script><h1>Hello</h1><p>One <a href='/x'>link</a> "
        b"and <code>code</code>.</p><ul><li>a</li><li>b</li></ul>"
        b"<pre>x = 1\n y = 2</pre></body></html>")


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path
        if path == "/page":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/text":
            self._send(200, b"plain words", "text/plain")
        elif path == "/big":
            self._send(200, b"a" * 200_000, "text/plain")
        elif path == "/missing":
            self._send(404, b"nope", "text/plain")
        elif path == "/hop":
            self._redirect("/page")
        elif path == "/away":
            self._redirect("http://localhost:%d/page" % self.server.server_port)
        elif path == "/ftp":
            self._redirect("ftp://example.com/x")
        elif path == "/image":                  # a PNG sent as a "page"
            self._send(200, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" * 50,
                       "text/html")
        elif path == "/pdf":
            self._send(200, b"%PDF-1.7\n\x00\x01\x02 stream", "application/pdf")
        elif path == "/slow":
            threading.Event().wait(3)
            self._send(200, b"late", "text/plain")

    def _send(self, code, body, kind):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, where):
        self.send_response(302)
        self.send_header("Location", where)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def site():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_web_fetch_turns_html_into_readable_text(site):
    out = web_fetch_impl(site + "/page")
    assert isinstance(out, Shown) and not _is_error(out)
    assert out.startswith("# Docs") or "# Hello" in out
    assert "# Hello" in out and "evil" not in out and "p{}" not in out
    assert f"[link]({site}/x)" in out and "`code`" in out
    assert "- a" in out and "- b" in out and "x = 1" in out
    assert out.display["kind"] == "web_fetch"
    assert out.display["content_type"].startswith("text/html")


def test_web_fetch_plain_text_redirect_inside_the_site_and_truncation(site):
    assert web_fetch_impl(site + "/text") == "plain words"
    assert "# Hello" in web_fetch_impl(site + "/hop")
    big = web_fetch_impl(site + "/big")
    assert big.display["was_truncated"] is True
    assert big.endswith("[Content truncated due to size limit]")
    assert len(big.encode()) < 125_000


def test_web_fetch_refuses_what_it_should_with_one_sentence(site):
    assert "HTTP error 404" in web_fetch_impl(site + "/missing")
    away = web_fetch_impl(site + "/away")         # 127.0.0.1 -> localhost
    assert _is_error(away) and "another site" in away and "localhost" in away
    assert "not http or https" in web_fetch_impl(site + "/ftp")
    assert "scheme" in web_fetch_impl("file:///etc/passwd")
    assert "scheme" in web_fetch_impl("ftp://example.com/x")
    assert _is_error(web_fetch_impl("  "))
    assert "positive" in web_fetch_impl(site + "/text", timeout=-1)


def test_web_fetch_refuses_a_binary_file_instead_of_reading_it_as_text(site):
    """A PDF or an image was decoded as text and up to 120 kB of noise was
    handed to the model. Checked on the bytes, like read_file, because
    servers mislabel files (the PNG here says text/html)."""
    for path in ("/image", "/pdf"):
        out = web_fetch_impl(site + path)
        assert _is_error(out) and "binary" in out, out
        assert len(out) < 300


def test_web_fetch_times_out(site):
    out = web_fetch_impl(site + "/slow", timeout=1)
    assert _is_error(out) and "timed out" in out


def test_web_fetch_with_no_scheme_assumes_https(monkeypatch):
    from ovat.tools import web_fetch
    assert web_fetch._normalize("example.com/a") == "https://example.com/a"
    assert web_fetch._normalize("//example.com") == "https://example.com"
    assert web_fetch._normalize("http://example.com") == "http://example.com"


def test_html_to_text_survives_broken_html():
    assert "Hello" in html_to_text("<p>Hello <b>wor<ld</p><div")
    assert html_to_text("") == ""


# ---- web_search -----------------------------------------------------------

class _Fake:
    """Replaces ws._request: records the call and returns canned JSON."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, body))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_searxng_asks_for_json_and_reads_results(monkeypatch):
    fake = _Fake({"results": [{"title": "T1", "url": "https://a.org",
                               "content": "about a"}]})
    monkeypatch.setattr(ws, "_request", fake)
    out = ws.web_search_impl("open vino", "searxng", "http://localhost:8080/")
    method, url, headers, body = fake.calls[0]
    assert method == "GET" and body is None
    assert url == "http://localhost:8080/search?q=open+vino&format=json"
    assert "T1" in out and "https://a.org" in out and "about a" in out
    assert out.display["sources"] == [{"title": "T1", "url": "https://a.org"}]


def test_tavily_posts_with_a_bearer_key(monkeypatch):
    monkeypatch.setenv("MY_TAVILY", "tvly-secret")
    fake = _Fake({"answer": "42", "results": [
        {"title": "T", "url": "https://t.io", "content": "c"}]})
    monkeypatch.setattr(ws, "_request", fake)
    out = ws.web_search_impl("q", "tavily", None, "MY_TAVILY")
    method, url, headers, body = fake.calls[0]
    assert (method, url) == ("POST", "https://api.tavily.com/search")
    assert headers == {"Authorization": "Bearer tvly-secret"}
    assert body == {"query": "q", "max_results": 5, "include_answer": True}
    assert out.startswith("Answer: 42") and "https://t.io" in out
    assert "tvly-secret" not in out


def test_serper_posts_with_its_key_header(monkeypatch):
    monkeypatch.setenv("MY_SERPER", "sk")
    fake = _Fake({"answerBox": {"answer": "yes"}, "organic": [
        {"title": "S", "link": "https://s.com", "snippet": "snip"}]})
    monkeypatch.setattr(ws, "_request", fake)
    out = ws.web_search_impl("q", "serper", None, "MY_SERPER")
    method, url, headers, body = fake.calls[0]
    assert (method, url) == ("POST", "https://google.serper.dev/search")
    assert headers == {"X-API-KEY": "sk"} and body == {"q": "q", "num": 5}
    assert "Answer: yes" in out and "https://s.com" in out


def test_brave_gets_with_its_token_header(monkeypatch):
    monkeypatch.setenv("MY_BRAVE", "bk")
    fake = _Fake({"web": {"results": [
        {"title": "B", "url": "https://b.net", "description": "desc"}]}})
    monkeypatch.setattr(ws, "_request", fake)
    out = ws.web_search_impl("a b", "brave", None, "MY_BRAVE")
    method, url, headers, body = fake.calls[0]
    assert method == "GET" and body is None
    assert url == ("https://api.search.brave.com/res/v1/web/search"
                   "?q=a%20b&count=5")
    assert headers == {"X-Subscription-Token": "bk"}
    assert "https://b.net" in out and "desc" in out


@pytest.mark.parametrize("provider, base_url, key_env, needle", [
    (None, None, None, "not set up"),
    ("tavily", None, None, "api_key_env"),
    ("tavily", None, "NOT_SET_ANYWHERE_XYZ", "NOT_SET_ANYWHERE_XYZ"),
    ("searxng", None, None, "base_url"),
    ("searxng", "ftp://x", None, "http"),
    ("searxng", "http://[broken", None, "http"),
    ("searxng", "http://x:bad", None, "http"),
])
def test_unconfigured_search_is_one_plain_sentence(
        monkeypatch, provider, base_url, key_env, needle):
    monkeypatch.delenv("NOT_SET_ANYWHERE_XYZ", raising=False)
    monkeypatch.setattr(ws, "_request", _Fake(RuntimeError("no network!")))
    out = ws.web_search_impl("q", provider, base_url, key_env)
    assert _is_error(out) and needle in out and "\n" not in out
    if provider != "searxng" or not base_url:
        assert "web_fetch works without a key" in out


def test_search_failures_are_sentences_and_never_show_the_key(monkeypatch):
    monkeypatch.setenv("MY_KEY", "supersecret")
    err = urllib.error.HTTPError("https://x", 401, "Unauthorized", {}, None)
    monkeypatch.setattr(ws, "_request", _Fake(err))
    out = ws.web_search_impl("q", "brave", None, "MY_KEY")
    assert "HTTP 401" in out and "MY_KEY" in out and "supersecret" not in out
    monkeypatch.setattr(ws, "_request", _Fake(urllib.error.URLError("down")))
    assert "could not reach brave" in ws.web_search_impl(
        "q", "brave", None, "MY_KEY")
    monkeypatch.setattr(ws, "_request", _Fake(ValueError("not json")))
    assert "not readable JSON" in ws.web_search_impl("q", "brave", None, "MY_KEY")
    forbidden = urllib.error.HTTPError("https://x", 403, "Forbidden", {}, None)
    monkeypatch.setattr(ws, "_request", _Fake(forbidden))
    assert "search.formats" in ws.web_search_impl(
        "q", "searxng", "http://localhost:8080")


def test_search_with_no_results_says_so(monkeypatch):
    monkeypatch.setattr(ws, "_request", _Fake({"results": []}))
    out = ws.web_search_impl("zzz", "searxng", "http://localhost:8080")
    assert out == "No results for: zzz"


def test_the_configured_entry_reaches_the_tool(monkeypatch):
    monkeypatch.setenv("MY_KEY", "k")
    fake = _Fake({"results": []})
    monkeypatch.setattr(ws, "_request", fake)
    cfg = WorkflowConfig.model_validate({
        "model": {"name": "m"}, "tools": [{
            "name": "web_search", "provider": "tavily",
            "api_key_env": "MY_KEY"}]})
    tools = factory.build_tools(cfg, events=EventBus())
    tools["web_search"]["function"](query="hi")
    assert fake.calls[0][1] == "https://api.tavily.com/search"


# ---- permissions, A2A and config -------------------------------------------

def test_default_permissions_of_the_new_tools():
    assert default_permission("builtin", name="web_fetch") == "ask"
    assert default_permission("builtin", name="web_search") == "ask"
    assert default_permission("builtin", name="todo") == "allow"
    assert default_permission("builtin", name="ask_user") == "allow"


def test_a_web_tool_with_nobody_to_approve_does_not_run():
    ran = []
    bus = EventBus()
    wrapped = observed("web_fetch", lambda url: ran.append(url), bus, "ask")
    assert _is_error(wrapped(url="https://example.com"))
    assert ran == []


def test_a2a_never_offers_the_web_tools_or_ask_user():
    tools = {n: {"kind": "builtin"} for n in
             ("search_docs", "todo", "ask_user", "web_fetch", "web_search")}
    result = exposure.decide(tools, None)
    assert set(result.exposed) == {"search_docs", "todo"}
    for name in ("ask_user", "web_fetch", "web_search"):
        assert "never offered" in result.hidden[name]
        with pytest.raises(ValueError, match="never offered"):
            exposure.decide(tools, [name])


def test_search_options_belong_to_web_search_only():
    base = {"model": {"name": "m"}}
    ok = WorkflowConfig.model_validate({**base, "tools": [{
        "name": "web_search", "provider": "brave",
        "api_key_env": "BRAVE_API_KEY"}]})
    assert ok.tools[0].provider == "brave"
    for bad in ({"name": "shell", "provider": "brave"},
                {"name": "web_fetch", "base_url": "http://x"},
                {"name": "web_search", "provider": "bing"},
                {"name": "web_search", "provider": "brave",
                 "api_key_env": "sk-abc123-secret"}):
        with pytest.raises(ValueError):
            WorkflowConfig.model_validate({**base, "tools": [bad]})
    # a web_search with no provider still loads: the tool explains itself
    WorkflowConfig.model_validate({**base, "tools": [{"name": "web_search"}]})


# ---- project context --------------------------------------------------------

def _cfg(**agent):
    return WorkflowConfig.model_validate({
        "model": {"name": "m"}, "agent": agent})


def test_project_context_is_off_by_default(tmp_path):
    (tmp_path / "AGENTS.md").write_text("secret rules")
    assert factory.compose_system_prompt(
        _cfg(system_prompt="base"), tmp_path) == "base"


def test_project_context_appends_agents_and_ovat_md(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Use tabs.")
    (tmp_path / "OVAT.md").write_text("Run make test.")
    prompt = factory.compose_system_prompt(
        _cfg(system_prompt="base", project_context=True), tmp_path)
    assert prompt.startswith("base\n\n")
    assert "Project instructions from AGENTS.md" in prompt
    assert "Use tabs." in prompt and "Run make test." in prompt
    assert prompt.index("AGENTS.md") < prompt.index("OVAT.md")


def test_project_context_with_no_files_or_no_base_prompt(tmp_path):
    assert factory.compose_system_prompt(
        _cfg(system_prompt="base", project_context=True), tmp_path) == "base"
    (tmp_path / "AGENTS.md").write_text("rules")
    prompt = factory.compose_system_prompt(
        _cfg(project_context=True), tmp_path)
    assert prompt.startswith("## Project instructions from AGENTS.md")
    (tmp_path / "AGENTS.md").write_text("   \n")        # empty is skipped
    assert factory.compose_system_prompt(
        _cfg(project_context=True), tmp_path) is None


def test_project_context_is_cut_and_says_so(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 20_000)
    prompt = factory.compose_system_prompt(
        _cfg(project_context=True), tmp_path)
    assert prompt.count("x") == factory.PROJECT_CONTEXT_CHARS
    assert "AGENTS.md was cut at 12000 characters" in prompt


def test_build_agent_hands_the_project_context_to_the_loop(
        tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("Always run make test.")
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(system_prompt="base", project_context=True)
    agent = factory.build_agent(cfg, skip_rag=True)
    system = agent.session.messages[0]["content"]
    assert system.startswith("base") and "Always run make test." in system


# ---- the shipped coding example ---------------------------------------------

def test_the_coding_example_is_what_it_says():
    from ovat.config.workflow import load_workflow
    cfg = load_workflow("examples/coding/workflow.yml")
    assert cfg.model.name == "Qwen3.6-35B-A3B-int4-ov"
    assert cfg.model.device == "GPU"
    assert cfg.model.tool_parser == "qwen3coder"
    assert cfg.model.source_model == "OpenVINO/Qwen3.6-35B-A3B-int4-ov"
    assert cfg.model.model_repository_path == "models"
    assert cfg.agent.project_context is True
    names = [t.name for t in cfg.tools]
    for wanted in ("read_file", "write_file", "edit", "grep", "shell",
                   "todo", "ask_user", "web_fetch"):
        assert wanted in names
    assert "web_search" not in names              # commented out on purpose
    # every tool it names exists, so the agent builds
    for tool in cfg.tools:
        assert tool.name in factory.BUILTIN_TOOL_SCHEMAS


def test_web_fetch_invalid_urls_are_tool_errors():
    for url in ("http://example.com:bad/x", "http://[broken", "http://"):
        assert _is_error(web_fetch_impl(url))


def test_embedded_void_html_does_not_hide_the_rest_of_the_page():
    assert "Keep this" in html_to_text('<embed src="x"><p>Keep this</p>')


@pytest.mark.parametrize("reply", [[], {"results": [None]}, {"results": 4}])
def test_search_malformed_results_are_tool_errors(monkeypatch, reply):
    monkeypatch.setattr(ws, "_request", _Fake(reply))
    assert _is_error(ws.web_search_impl("q", "searxng", "http://localhost"))


def test_search_network_error_does_not_echo_provider_secrets(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret-key")
    monkeypatch.setattr(ws, "_request", _Fake(
        urllib.error.URLError("request with secret-key failed")))
    assert "secret-key" not in ws.web_search_impl("q", "brave", None, "MY_KEY")


def test_search_http_adapter_caps_the_reply_and_refuses_redirects(site, monkeypatch):
    # A redirect would otherwise forward X-API-KEY to an unapproved host.
    with pytest.raises(urllib.error.HTTPError) as error:
        ws._request("GET", site + "/away", {"X-API-KEY": "secret"}, None)
    assert error.value.code == 302
    monkeypatch.setattr(ws, "MAX_RESPONSE_BYTES", 10)
    with pytest.raises(ValueError, match="too large"):
        ws._request("GET", site + "/big", {}, None)


def test_all_helper_tool_arguments_derive_from_the_schema():
    from ovat.agent.arg_models import args_model_from_schema
    samples = {
        "todo": {"action": "write", "todos": [_item(1)]},
        "ask_user": {"question": "Which?", "choices": ["a", "b"]},
        "web_fetch": {"url": "https://example.com"},
        "web_search": {"query": "docs"},
    }
    for name, args in samples.items():
        model = args_model_from_schema(name, factory.BUILTIN_TOOL_SCHEMAS[name])
        parsed = model.model_validate(args).model_dump(exclude_none=True)
        assert parsed == args
    item = factory.BUILTIN_TOOL_SCHEMAS["todo"]["function"]["parameters"][
        "properties"]["todos"]["items"]
    assert set(item["properties"]) == {"id", "content", "status", "priority"}


def test_web_fetch_blocks_cloud_metadata_private_ips_and_dns_rebinding(
        monkeypatch):
    import socket
    from ovat.tools import web_fetch

    for blocked in (
        "http://169.254.169.254/latest/meta-data/",
        "http://[fd00:ec2::254]/latest/meta-data/",
        "http://10.0.0.1/internal",
        "http://192.168.1.1/admin",
        "http://172.16.0.5/secret",
        "http://0.0.0.0:8080/",
    ):
        out = web_fetch_impl(blocked, timeout=1)
        assert _is_error(out) and "refusing to fetch" in out, (blocked, out)

    # A domain that resolves via DNS rebinding to 127.0.0.1 must be refused
    # even though literal http://127.0.0.1 is allowed for local dev servers.
    monkeypatch.setattr(
        web_fetch.socket,
        "getaddrinfo",
        lambda host, *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "",
                                 ("127.0.0.1", 0))],
    )
    rebound = web_fetch_impl("http://rebind.evil.example/steal")
    assert _is_error(rebound) and "loopback" in rebound


def test_web_search_searxng_blocks_cloud_metadata_base_url():
    out = ws.web_search_impl(
        "query", "searxng", "http://169.254.169.254/latest/meta-data"
    )
    assert _is_error(out) and "metadata" in out

