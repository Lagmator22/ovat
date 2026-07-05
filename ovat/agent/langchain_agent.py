# ovat/agent/langchain_agent.py
"""Layer 3 (alternate engine): run the same agent through LangChain.

The proposal names LangChain as the primary framework integration, so this is
the `agent.type: react` path. The trick is that the rest of OVAT must not care
which engine runs. AgentLoop exposes .run(text) -> text; so does LangChainAgent
below. The factory hands back one or the other, and the CLI calls .run() either
way. That is the same polymorphism the providers use, applied one layer up.

Under the hood I use LangChain's create_agent (its v1 tool-calling agent) with
a ChatOpenAI model pointed at the OVMS /v3 endpoint. OVMS is OpenAI-compatible
and decodes tool calls with --tool_parser, so LangChain's native tool calling
works against it without any special glue.

I import langchain lazily inside the build function. That keeps `import ovat`
cheap and lets someone who only uses the native loop skip the heavy install.
"""
from pydantic import Field, create_model

from ovat.config.workflow import WorkflowConfig

# JSON-schema type names -> Python types, for deriving argument models.
_JSON_TO_PY = {"string": str, "integer": int, "number": float,
               "boolean": bool, "array": list, "object": dict}


def _args_model_from_schema(name: str, schema: dict):
    """Derive a pydantic argument model from a tool's OpenAI SCHEMA.

    Design note: each tool already ships its full contract in the co-located
    SCHEMA dict (that was the whole point of co-locating it). Deriving the
    LangChain model from it means ONE source of truth: the old hand-kept
    _ARGS_MODELS registry meant every new tool worked on the native path and
    then crashed on react until someone remembered the second list.

    create_model is pydantic's runtime class factory: it builds the same kind
    of class a `class ...Args(BaseModel)` block would, from data instead of
    source code.
    """
    params = schema["function"].get("parameters", {})
    required = set(params.get("required", []))
    fields = {}
    for pname, spec in params.get("properties", {}).items():
        py_type = _JSON_TO_PY.get(spec.get("type"), str)
        # Required fields use `...` (pydantic's "no default, must be given");
        # optional ones take the schema's default (None if it names none).
        default = ... if pname in required else spec.get("default", None)
        fields[pname] = (py_type,
                         Field(default, description=spec.get("description", "")))
    return create_model(f"{name}_Args", **fields)


def _wrap_tools(tools: dict) -> list:
    """Turn my {name: {schema, function}} dict into LangChain StructuredTools.

    Each wrapped tool reuses the exact same callable my native loop runs, so a
    tool behaves identically no matter which engine called it. The retriever I
    bound into search_docs in the factory is already inside that callable.
    """
    from langchain_core.tools import StructuredTool

    wrapped = []
    for name, spec in tools.items():
        wrapped.append(StructuredTool.from_function(
            func=spec["function"],
            name=name,
            description=spec["schema"]["function"]["description"],
            args_schema=_args_model_from_schema(name, spec["schema"]),
        ))
    return wrapped


def _build_chat_model(config: WorkflowConfig):
    """Build a ChatOpenAI pointed at OVMS. Construction makes no network call."""
    from langchain_openai import ChatOpenAI

    m = config.model
    # api_key is required by the SDK but ignored by OVMS; any string works.
    # temperature 0 keeps the demo answers stable across runs.
    return ChatOpenAI(base_url=m.ovms_url, api_key="not-needed",
                      model=m.name, temperature=0)


class LangChainAgent:
    """Adapter so a LangChain agent looks exactly like my native AgentLoop."""

    def __init__(self, graph, tools: dict, max_iterations: int,
                 system_prompt: str | None):
        self._graph = graph
        # I keep the original tools dict so `ovat run --dry-run` can print the
        # tool names the same way it does for the native loop.
        self.tools = tools
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        # langgraph counts each node visit, and one tool round is roughly two
        # visits plus the final answer. This keeps my cap close in meaning to
        # the native loop's max_iterations.
        self._recursion_limit = max_iterations * 2 + 1

    def run(self, user_message: str) -> str:
        """Run the LangChain agent for one message and return the final text."""
        from langgraph.errors import GraphRecursionError

        try:
            result = self._graph.invoke(
                {"messages": [("user", user_message)]},
                config={"recursion_limit": self._recursion_limit},
            )
        except GraphRecursionError:
            # Same wording as the native loop so the two engines fail alike.
            return (f"Error: I reached my max of {self.max_iterations} steps "
                    f"without a final answer.")
        final = result["messages"][-1]
        return getattr(final, "content", str(final))


def build_react_agent(config: WorkflowConfig, tools: dict,
                      llm=None) -> LangChainAgent:
    """Build the LangChain react agent. `llm` is injectable for testing.

    In production I build a ChatOpenAI against OVMS. In tests I pass a fake chat
    model so the whole tool-calling loop runs on any machine with no server.
    """
    try:
        from langchain.agents import create_agent
    except ImportError as exc:
        raise RuntimeError(
            "agent.type 'react' needs LangChain. Install it with: "
            "pip install 'ovat[langchain]'"
        ) from exc

    chat = llm if llm is not None else _build_chat_model(config)
    kwargs = {}
    if config.agent.system_prompt:
        kwargs["system_prompt"] = config.agent.system_prompt
    graph = create_agent(chat, _wrap_tools(tools), **kwargs)
    return LangChainAgent(graph, tools, config.agent.max_iterations,
                          config.agent.system_prompt)
