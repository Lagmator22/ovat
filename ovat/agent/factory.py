# ovat/agent/factory.py
"""Layer 2: the component factory, config in, ready-to-run agent out.

Design note: this is where the whole architecture pays off. The factory reads
a validated WorkflowConfig and builds the real objects: the LLM provider, the
tools, the optional RAG retriever, and the agent itself, all wired together.
The user never constructs anything by hand; they write YAML, the factory does
the rest.

This is also where the ABC bet from base.py cashes in. The config names a
provider as a STRING (genai vs ovms, sqlite-vec vs a future backend) and the
factory maps that string to the matching concrete class. Swapping a backend is
a one-line YAML edit, not a code change.
"""
import copy
import logging
import os

from ovat.agent.events import EventBus, observed
from ovat.agent.loop import AgentLoop
from ovat.agent.permissions import default_permission
from ovat.config.workflow import WorkflowConfig
from ovat.providers.backend import LLMBackend
from ovat.providers.base import (EmbeddingsProvider, LLMProvider,
                                 RetrieverProvider)
from ovat.providers.llm_ovms import OVMSLLMProvider
from ovat.tools import describe_image as describe_image_tool
from ovat.tools import search_docs as search_docs_tool
from ovat.tools import transcribe as transcribe_tool


log = logging.getLogger(__name__)


# The schema (the menu the model reads) for each built-in tool. I keep the
# function-building separate, below, because search_docs needs the retriever
# bound in while the others do not.
BUILTIN_TOOL_SCHEMAS = {
    "search_docs": search_docs_tool.SCHEMA,
    "transcribe": transcribe_tool.SCHEMA,
    "describe_image": describe_image_tool.SCHEMA,
}


def build_llm(config: WorkflowConfig) -> LLMProvider:
    """I build the LLM provider from the model section of the config.

    The return type is the ABC, not a concrete class, and that matters more
    than it looks. This used to be annotated `-> OVMSLLMProvider` and hardcode
    that one plug, which meant GenAILLMProvider honoured the contract and was
    unreachable from build_agent. A factory whose signature names a concrete
    class has stopped using its own abstraction; the annotation is the tell.

    Design note: constructing either provider does NOT connect or load, it
    only sets up the client. So this is safe to call without OVMS running,
    which is why the factory tests pass on the Mac.
    """
    m = config.model
    if m.provider == "genai":
        # Imported lazily, exactly like build_embedder does: the OVMS path
        # must not pay openvino_genai's import cost.
        from ovat.providers.llm_genai import GenAILLMProvider
        # The same ceiling the OVMS path takes. GenAILLMProvider's own 256
        # default cut every agent answer short, and the "uncapped" local chat
        # does not apply here: that chat streams and Esc stops it, while an
        # agent turn is one blocking call nobody can interrupt mid-generation.
        return GenAILLMProvider(m.name, m.device,
                                max_new_tokens=m.max_tokens,
                                enable_thinking=m.enable_thinking)
    if m.provider != "ovms":
        raise ValueError(
            f"Unknown model provider '{m.provider}'. Supported: ovms, genai."
        )
    backend = LLMBackend.from_config(config)
    return OVMSLLMProvider(base_url=backend.url, model=backend.model,
                           timeout=backend.timeout,
                           temperature=backend.temperature,
                           max_tokens=backend.max_tokens,
                           sampling=backend.openai_kwargs(),
                           extra_body=backend.extra_body(),
                           stream=(m.device.upper() != "NPU"),
                           api_key=backend.api_key)


def build_embedder(config: WorkflowConfig) -> EmbeddingsProvider:
    """Pick and build the embedder named in config.rag.embeddings.provider.

    This is the ABC swap in action: the string decides the class. I import the
    concrete provider lazily so that merely importing the factory (or running a
    test that never touches RAG) does not drag in openvino_genai.
    """
    if config.rag is None:
        # A real exception with a real message, not `assert`. An AssertionError
        # carries no text -- str(AssertionError()) is "" -- and every caller
        # wraps this in `except Exception`, so an assert here reaches the user
        # as "Could not build the retriever: " with nothing after the colon.
        # That is worse than the traceback it was meant to prevent: a blank
        # error tells you neither what happened nor what to do. Asserts are
        # also stripped by `python -O`, so the guarantee evaporates in exactly
        # the build where you would want it most.
        raise ValueError(
            "this workflow has no rag: section, so there is no embedder to "
            "build. Add one, or call build_rag(), which returns None here "
            "rather than raising.")
    emb = config.rag.embeddings
    if emb.provider == "genai":
        from ovat.providers.embeddings_genai import GenAIEmbeddingsProvider
        # Check for the folder BEFORE handing the path to openvino_genai. A
        # missing model there surfaces as a C++ assertion out of core.cpp
        # ("Check 'util::directory_exists...'"), which the CLI then rendered as
        # a traceback -- while `ovat doctor` was already reporting the same
        # thing as a polite warn row. Errors are for users: same fact, same
        # wording, one sentence.
        if not os.path.isdir(emb.model):
            raise RuntimeError(
                f"embeddings model not found at {emb.model!r}. Export it once "
                f"with:\n"
                f"  pip install \"ovat[convert]\"\n"
                f"  optimum-cli export openvino --model BAAI/bge-small-en-v1.5 "
                f"--task feature-extraction {emb.model}\n"
                f"then run 'ovat index <folder> <config>'. "
                f"'ovat doctor <config>' reports this too.")
        return GenAIEmbeddingsProvider(model_path=emb.model, device=emb.device)
    if emb.provider == "ovms":
        from ovat.providers.embeddings_ovms import OVMSEmbeddingsProvider
        # The server path reuses the same OVMS url (and timeout) the LLM uses.
        return OVMSEmbeddingsProvider(base_url=config.model.ovms_url, model=emb.model,
                                      timeout=config.model.request_timeout,
                                      api_key=config.model.ovms_api_key()
                                      or "not-needed")
    raise ValueError(
        f"Unknown embeddings provider '{emb.provider}'. Supported: genai, ovms."
    )


def build_retriever(config: WorkflowConfig,
                    embedder: EmbeddingsProvider) -> RetrieverProvider:
    """Pick and build the vector store named in config.rag.retriever.provider.

    Takes the embedder as an argument (composition) so a test can hand in a fake
    embedder and exercise the whole store without loading a real model.
    """
    if config.rag is None:
        # A real exception, not `assert`, for the reason build_embedder gives.
        raise ValueError(
            "this workflow has no rag: section, so there is no retriever to "
            "build. Add one, or call build_rag(), which returns None here "
            "rather than raising.")
    ret = config.rag.retriever
    if ret.provider in ("sqlite-vec", "sqlite_vec"):
        from ovat.providers.retriever_sqlitevec import SQLiteVecRetrieverProvider
        return SQLiteVecRetrieverProvider(
            embedder=embedder,
            dim=config.rag.embeddings.dim,
            db_path=ret.db_path,
        )
    if ret.provider in ("memory", "in-memory", "in_memory"):
        from ovat.providers.retriever_memory import InMemoryRetrieverProvider
        # db_path is deliberately ignored rather than rejected: a config that
        # switches backend to try something out should not also have to delete
        # a line that now means nothing.
        return InMemoryRetrieverProvider(
            embedder=embedder,
            dim=config.rag.embeddings.dim,
        )
    raise ValueError(
        f"Unknown retriever provider '{ret.provider}'. "
        f"Supported: sqlite-vec, memory."
    )


def build_rag(config: WorkflowConfig) -> RetrieverProvider | None:
    """Build the full RAG retriever, or None when no rag section is configured.

    Returning None on purpose: a workflow without a `rag:` block keeps
    search_docs in stub mode, which is exactly what the wiring tests rely on.
    """
    if config.rag is None:
        return None
    embedder = build_embedder(config)
    return build_retriever(config, embedder)


def _make_search_docs(retriever: RetrieverProvider | None):
    """Bind the retriever into the search_docs callable the loop will run.

    When retriever is None the tool returns its obvious stub. When it is a real
    retriever the same tool returns real chunks with citations. The agent loop
    cannot tell the difference; only the wiring here changed.
    """
    return lambda query, top_k=5: search_docs_tool.search_docs_impl(
        query, top_k, retriever
    )


def _make_transcribe(tool_cfg=None):
    """Bind this tool's configured model and device into the callable.

    Default-arg binding, for the same reason _make_mcp_caller uses it: the
    loop calls these with the MODEL's arguments only, so anything from the
    config has to be captured here rather than looked up later.
    """
    model = getattr(tool_cfg, "model", None)
    device = getattr(tool_cfg, "device", None)
    return lambda path, language="en", _m=model, _d=device: (
        transcribe_tool.transcribe_impl(path, language,
                                        model=_m, device=_d))


def _make_describe_image(tool_cfg=None):
    """Same binding as _make_transcribe, for the vision model."""
    model = getattr(tool_cfg, "model", None)
    device = getattr(tool_cfg, "device", None)
    return lambda image_path, prompt="Describe this image in one paragraph.", \
        _m=model, _d=device: (
            describe_image_tool.describe_image_impl(image_path, prompt,
                                                    model=_m, device=_d))


def _make_mcp_caller(server, tool_name: str):
    """Bind one remote MCP tool into a plain callable for the loop.

    Default-arg binding on purpose: a lambda closing over loop variables would
    capture the LAST tool of the server, the classic late-binding trap.
    """
    return lambda _server=server, _name=tool_name, **kwargs: (
        _server.call_tool(_name, kwargs)
    )


def _make_a2a_tool(tool_cfg, servers: list | None, events: EventBus | None):
    """A remote A2A agent as one {schema, function} tool.

    The SDK is imported here and only here on the build path, so a workflow
    without `type: a2a` never needs the [a2a] extra. One unreachable peer
    must not stop the run from starting (unlike a broken MCP command, it is
    another machine that may come back): its card is read with a short
    budget, a miss is logged, and the first call tries again.
    """
    from ovat.a2a.protocol import PeerError
    from ovat.a2a.sdk import require_sdk
    from ovat.tools import a2a_call
    require_sdk()
    from ovat.a2a.sdk.client import SDKPeer

    token = os.environ.get(tool_cfg.token_env, "").strip() \
        if tool_cfg.token_env else ""
    peer = SDKPeer(tool_cfg.url, token=token or None,
                   timeout=tool_cfg.timeout or 900.0)
    if servers is not None:
        servers.append(peer)           # close_agent() closes it
    card = None
    try:
        card = peer.card()
    except PeerError as exc:
        log.warning("remote agent %s at %s: %s (%s); will retry on first "
                    "use", tool_cfg.name, tool_cfg.url, exc.kind, exc.detail)
    schema = copy.deepcopy(a2a_call.SCHEMA)
    schema["function"]["name"] = tool_cfg.name
    schema["function"]["description"] = (
        tool_cfg.description
        or a2a_call.describe(tool_cfg.name, tool_cfg.url, card))
    missing = bool(tool_cfg.token_env) and not token
    return {
        "schema": schema,
        "function": lambda message, _p=peer, _c=tool_cfg, _m=missing: (
            a2a_call.a2a_call_impl(message, _p, _c.name, _c.url, events,
                                   _c.token_env, _m)),
        "kind": "a2a",
    }


def close_agent(agent) -> None:
    """Shut down anything the agent spawned. Safe to call twice, or never.

    `mcp_servers` holds everything with a close(): MCP servers and A2A
    peers (each owns a thread and a connection). Of the MCP side, only
    mcp_stdio tools own a real OS resource: each launches a subprocess
    with its own event-loop thread. Nothing used to own them for shutdown, so
    they lived until the interpreter exited via MCPStdioServer's atexit hook.
    For `ovat run` that is fine. For the TUI, where every `/engine` switch
    rebuilds the agent, and for `ovat connect` serving many requests, the
    subprocesses accumulate.

    A free function rather than a method because the four engine adapters are
    different classes and none of them should have to learn about MCP.
    """
    for server in getattr(agent, "mcp_servers", ()) or ():
        try:
            server.close()
        except Exception:
            # Shutting down must not raise: one wedged server should not stop
            # the others being closed, nor take down the caller.
            pass


def build_tools(config: WorkflowConfig,
                retriever: RetrieverProvider | None = None,
                servers: list | None = None,
                events: EventBus | None = None) -> dict:
    """I turn the list of tool configs into the dict my loop expects.

    Two tool types:
    - builtin: my own tools, called in-process (the fast path).
    - mcp_stdio: launch `command` as an MCP server subprocess and import
      EVERY tool it advertises. This is what makes workflow.yml an open
      socket: any third-party MCP server plugs in with two YAML lines.
    The loop consumes the same {schema, function} dict either way; it never
    learns which side of the wire a tool lives on. Each entry also records
    its `kind` (the config type it came from), which the engines ignore and
    the A2A server reads to decide what a remote caller may trigger.
    """
    # Each builder takes the tool's OWN config, so `model:` and `device:` in
    # workflow.yml reach the tool that declared them. search_docs ignores it;
    # its model is the embedder in the rag: section.
    builders = {
        "search_docs": lambda cfg: _make_search_docs(retriever),
        "transcribe": _make_transcribe,
        "describe_image": _make_describe_image,
    }
    tools = {}
    for tool_cfg in config.tools:
        if tool_cfg.type == "builtin":
            if tool_cfg.name not in BUILTIN_TOOL_SCHEMAS:
                raise ValueError(
                    f"Unknown builtin tool '{tool_cfg.name}'. "
                    f"Available: {list(BUILTIN_TOOL_SCHEMAS)}"
                )
            tools[tool_cfg.name] = {
                "schema": BUILTIN_TOOL_SCHEMAS[tool_cfg.name],
                "function": builders[tool_cfg.name](tool_cfg),
                "kind": "builtin",
                "_entry": tool_cfg,
            }
        elif tool_cfg.type == "mcp_stdio":
            if not tool_cfg.command:
                raise ValueError(
                    f"Tool '{tool_cfg.name}' has type mcp_stdio but no "
                    f"command to launch the server with."
                )
            from ovat.tools.mcp_client import (MCPStdioServer,
                                               openai_schema_from_mcp_tool)
            server = MCPStdioServer(tool_cfg.command, env=tool_cfg.env)
            if servers is not None:
                # Handed back so somebody owns the subprocess's lifetime.
                servers.append(server)
            for remote in server.tools:
                tools[remote.name] = {
                    "schema": openai_schema_from_mcp_tool(remote),
                    "function": _make_mcp_caller(server, remote.name),
                    "kind": "mcp_stdio",
                    "_entry": tool_cfg,
                }
        elif tool_cfg.type == "a2a":
            tools[tool_cfg.name] = {**_make_a2a_tool(tool_cfg, servers, events),
                                    "_entry": tool_cfg}
        else:
            raise ValueError(
                f"Unsupported tool type '{tool_cfg.type}' for '{tool_cfg.name}'. "
                f"Supported: builtin, mcp_stdio, a2a."
            )
    # Every tool from THIS config entry gets its permission: an mcp_stdio
    # entry's applies to each tool its server advertises.
    for tool_cfg in config.tools:
        default = default_permission(tool_cfg.type, tool_cfg.url)
        for spec in tools.values():
            if spec.get("_entry") is tool_cfg:
                spec["permission"] = tool_cfg.permission or default
    for spec in tools.values():
        spec.pop("_entry", None)
        spec.setdefault("permission", "allow")
    if config.subagents:
        from ovat.agent.subagents import make_delegate_tool
        if "delegate" in tools:
            raise ValueError("a tool named 'delegate' clashes with the tool "
                             "subagents: adds; rename it.")
        tools["delegate"] = make_delegate_tool(
            config.subagents, events if events is not None else EventBus())
    return tools


def observe_tools(tools: dict, bus: EventBus) -> dict:
    """The same tools, each function wrapped to report to `bus`.

    Kept apart from build_tools on purpose: build_tools decides WHAT the
    tools are, this decides how they are watched. All four engines call
    tools[name]["function"], so this one step gives every engine tool events
    and a cancel check (see ovat/agent/events.py).
    """
    return {name: {**spec, "function": observed(
                name, spec["function"], bus,
                spec.get("permission", "allow"))}
            for name, spec in tools.items()}


# Every engine the factory can build. Named here so the error message, the
# config docs and the dispatch below cannot drift apart.
AGENT_TYPES = ("native", "react", "llamaindex", "openai-agents")


def build_agent(config: WorkflowConfig, skip_rag: bool = False,
                select_tools=None):
    """I assemble the full agent from a config: the one call the CLI makes.

    Dispatch on agent.type so the same YAML can pick my native loop or the
    LangChain front-end. An unknown type fails loudly here instead of silently
    falling back to the wrong engine.

    skip_rag is for `ovat run --dry-run`: building the real embedder loads a
    model off disk, which is a heavy backend in the same category as the OVMS
    server. Dry-run proves the wiring on any machine, so it skips that load and
    search_docs stays in stub mode for the preview.

    select_tools, when given, takes the built tools dict and returns the
    subset this agent may use. `ovat a2a serve` passes one, so a tool that
    is not exposed is not merely left off the Agent Card: the model is
    never shown it at all, and cannot be talked into calling it.
    """
    # Build RAG first so the in-process search_docs has its retriever.
    #
    # This does NOT reach a standalone MCP server, which the previous comment
    # here claimed. configure() sets a module-level slot in THIS process; a
    # `type: mcp_stdio` tool runs in a separate one and cannot see it. Such a
    # server builds its own retriever from --config; see
    # search_docs.configure_from_config.
    # Checked BEFORE anything is built. The framework engines each construct
    # their own client from a URL: LangChain a ChatOpenAI, LlamaIndex an
    # OpenAILike, the Agents SDK an AsyncOpenAI. None can consume an
    # in-process GenAILLMProvider, so the combination cannot work. Rejecting
    # it here means the user gets a sentence naming the fix, instead of
    # openvino_genai failing to find a model directory several steps later.
    if config.model.provider == "genai" and config.agent.type != "native":
        raise ValueError(
            f"model.provider 'genai' runs in-process and cannot be used with "
            f"agent.type '{config.agent.type}', which needs an "
            f"OpenAI-compatible URL. Use agent.type: native, or "
            f"model.provider: ovms."
        )
    retriever = None if skip_rag else build_rag(config)
    if retriever is not None:
        search_docs_tool.configure(retriever)

    llm = build_llm(config)
    # Collected so close_agent() can shut the subprocesses down. Without an
    # owner they survive until the interpreter exits.
    mcp_servers: list = []
    # One bus per agent: the TUI and the A2A server subscribe to it, and its
    # `stop` flag is how either of them cancels a run.
    events = EventBus()
    tools = build_tools(config, retriever=retriever, servers=mcp_servers,
                        events=events)
    if select_tools is not None:
        tools = select_tools(tools)
    tools = observe_tools(tools, events)

    agent_type = config.agent.type
    def _own(agent):
        """Hand the agent its subprocesses and its event bus."""
        agent.mcp_servers = mcp_servers
        agent.events = events
        return agent

    if agent_type == "native":
        return _own(AgentLoop(
            llm=llm,
            tools=tools,
            system_prompt=config.agent.system_prompt,
            max_iterations=config.agent.max_iterations,
            engine_name=agent_type,
            stop=events.stop,
            events=events,
        ))
    # Each framework is imported INSIDE its branch so the native path never
    # pays the import cost of a framework it does not use, and so a machine
    # with only one of them installed still runs the others' configs' errors
    # as readable text rather than an ImportError at module load.
    if agent_type == "react":
        from ovat.agent.langchain_agent import build_react_agent
        return _own(build_react_agent(config, tools))

    if agent_type == "llamaindex":
        from ovat.agent.llamaindex_agent import build_llamaindex_agent
        return _own(build_llamaindex_agent(config, tools))

    if agent_type == "openai-agents":
        from ovat.agent.openai_agents_agent import build_openai_agents_agent
        return _own(build_openai_agents_agent(config, tools))

    raise ValueError(
        f"Unknown agent type '{agent_type}'. Supported: "
        f"{', '.join(AGENT_TYPES)}."
    )
