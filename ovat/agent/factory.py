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
from ovat.agent.loop import AgentLoop
from ovat.config.workflow import WorkflowConfig
from ovat.providers.base import EmbeddingsProvider, RetrieverProvider
from ovat.providers.llm_ovms import OVMSLLMProvider
from ovat.tools import search_docs as search_docs_tool
from ovat.tools import transcribe as transcribe_tool


# The schema (the menu the model reads) for each built-in tool. I keep the
# function-building separate, below, because search_docs needs the retriever
# bound in while transcribe does not.
BUILTIN_TOOL_SCHEMAS = {
    "search_docs": search_docs_tool.SCHEMA,
    "transcribe": transcribe_tool.SCHEMA,
}


def build_llm(config: WorkflowConfig) -> OVMSLLMProvider:
    """I build the LLM provider from the model section of the config.

    Design note: constructing OVMSLLMProvider does NOT connect to the server,
    it only sets up the client. So this is safe to call without OVMS running,
    which is why the factory tests pass on the Mac.
    """
    m = config.model
    return OVMSLLMProvider(base_url=m.ovms_url, model=m.name)


def build_embedder(config: WorkflowConfig) -> EmbeddingsProvider:
    """Pick and build the embedder named in config.rag.embeddings.provider.

    This is the ABC swap in action: the string decides the class. I import the
    concrete provider lazily so that merely importing the factory (or running a
    test that never touches RAG) does not drag in openvino_genai.
    """
    emb = config.rag.embeddings
    if emb.provider == "genai":
        from ovat.providers.embeddings_genai import GenAIEmbeddingsProvider
        return GenAIEmbeddingsProvider(model_path=emb.model, device=emb.device)
    if emb.provider == "ovms":
        from ovat.providers.embeddings_ovms import OVMSEmbeddingsProvider
        # The server path reuses the same OVMS url the LLM talks to.
        return OVMSEmbeddingsProvider(base_url=config.model.ovms_url, model=emb.model)
    raise ValueError(
        f"Unknown embeddings provider '{emb.provider}'. Supported: genai, ovms."
    )


def build_retriever(config: WorkflowConfig,
                    embedder: EmbeddingsProvider) -> RetrieverProvider:
    """Pick and build the vector store named in config.rag.retriever.provider.

    Takes the embedder as an argument (composition) so a test can hand in a fake
    embedder and exercise the whole store without loading a real model.
    """
    ret = config.rag.retriever
    if ret.provider in ("sqlite-vec", "sqlite_vec"):
        from ovat.providers.retriever_sqlitevec import SQLiteVecRetrieverProvider
        return SQLiteVecRetrieverProvider(
            embedder=embedder,
            dim=config.rag.embeddings.dim,
            db_path=ret.db_path,
        )
    raise ValueError(
        f"Unknown retriever provider '{ret.provider}'. Supported: sqlite-vec."
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


def _make_transcribe():
    return lambda file_path, language="en": transcribe_tool.transcribe_impl(
        file_path, language
    )


def build_tools(config: WorkflowConfig,
                retriever: RetrieverProvider | None = None) -> dict:
    """I turn the list of tool configs into the dict my loop expects.

    Heads up: only built-in tools are supported right now. If the YAML names a
    tool I do not have, I raise a clear error instead of silently giving the
    agent an empty toolbox.
    """
    builders = {
        "search_docs": lambda: _make_search_docs(retriever),
        "transcribe": lambda: _make_transcribe(),
    }
    tools = {}
    for tool_cfg in config.tools:
        if tool_cfg.type != "builtin":
            raise ValueError(
                f"Unsupported tool type '{tool_cfg.type}' for '{tool_cfg.name}'. "
                f"Only 'builtin' is supported right now."
            )
        if tool_cfg.name not in BUILTIN_TOOL_SCHEMAS:
            raise ValueError(
                f"Unknown builtin tool '{tool_cfg.name}'. "
                f"Available: {list(BUILTIN_TOOL_SCHEMAS)}"
            )
        tools[tool_cfg.name] = {
            "schema": BUILTIN_TOOL_SCHEMAS[tool_cfg.name],
            "function": builders[tool_cfg.name](),
        }
    return tools


def build_agent(config: WorkflowConfig, skip_rag: bool = False):
    """I assemble the full agent from a config: the one call the CLI makes.

    Dispatch on agent.type so the same YAML can pick my native loop or the
    LangChain front-end. An unknown type fails loudly here instead of silently
    falling back to the wrong engine.

    skip_rag is for `ovat run --dry-run`: building the real embedder loads a
    model off disk, which is a heavy backend in the same category as the OVMS
    server. Dry-run proves the wiring on any machine, so it skips that load and
    search_docs stays in stub mode for the preview.
    """
    # Build RAG first so the same retriever is shared by the in-process tool and
    # the standalone MCP server (configure() sets the module-level slot too).
    retriever = None if skip_rag else build_rag(config)
    if retriever is not None:
        search_docs_tool.configure(retriever)

    llm = build_llm(config)
    tools = build_tools(config, retriever=retriever)

    agent_type = config.agent.type
    if agent_type == "native":
        return AgentLoop(
            llm=llm,
            tools=tools,
            system_prompt=config.agent.system_prompt,
            max_iterations=config.agent.max_iterations,
        )
    if agent_type == "react":
        # Imported here so the native path never pays the LangChain import cost.
        from ovat.agent.langchain_agent import build_react_agent
        return build_react_agent(config, tools)

    raise ValueError(
        f"Unknown agent type '{agent_type}'. Supported: native, react."
    )
