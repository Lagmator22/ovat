# ovat/agent/langchain_agent.py
"""LangChain ``react`` agent engine adapter.

Implements the same ``.run(text) -> str`` interface as ``AgentLoop`` so callers
and factories can swap engines transparently. Uses LangChain's v1
``create_agent`` graph with ``ChatOpenAI`` pointed at OVMS ``/v3``, importing
``langchain`` lazily so native-only users avoid that dependency cost.
"""
import time

from ovat.agent.arg_models import args_model_from_schema
from ovat.agent.usage import framework_trace
from ovat.providers.backend import LLMBackend
from ovat.config.workflow import WorkflowConfig


def _wrap_tools(tools: dict) -> list:
    """Wrap the normalized ``{name: {schema, function}}`` map into LangChain ``StructuredTool``s.

    Each wrapped tool reuses the underlying callable (and any bound retriever)
    directly so tool behavior matches the native loop.
    """
    from langchain_core.tools import StructuredTool

    wrapped = []
    for name, spec in tools.items():
        wrapped.append(StructuredTool.from_function(
            func=spec["function"],
            name=name,
            description=spec["schema"]["function"]["description"],
            args_schema=args_model_from_schema(name, spec["schema"]),
        ))
    return wrapped


def _usage_per_call(messages: list) -> list:
    """(prompt, completion) tokens for each model reply among `messages`.

    ChatOpenAI copies the server's `usage` block onto each AIMessage as
    usage_metadata (input_tokens / output_tokens), and leaves it None when
    the reply carried none. None stays None: absent is not zero.
    """
    calls = []
    for message in messages:
        if getattr(message, "type", None) != "ai":
            continue
        usage = getattr(message, "usage_metadata", None) or {}
        calls.append((usage.get("input_tokens"), usage.get("output_tokens")))
    return calls


def _build_chat_model(config: WorkflowConfig):
    """Build a ChatOpenAI pointed at OVMS. Construction makes no network call."""
    from langchain_openai import ChatOpenAI

    # One shared description of the connection, so this engine cannot drift
    # from the other three. See ovat/providers/backend.py for why that
    # matters: temperature was set here and in llamaindex, and omitted in the
    # other two, for a whole release.
    b = LLMBackend.from_config(config)
    return ChatOpenAI(base_url=b.url, api_key=b.api_key, model=b.model,
                      temperature=b.temperature, timeout=b.timeout,
                      max_tokens=b.max_tokens, **b.openai_kwargs(),
                      extra_body=b.extra_body() or None)


class LangChainAgent:
    """Adapter presenting a LangChain agent behind the ``AgentLoop`` interface."""

    def __init__(self, graph, tools: dict, max_iterations: int,
                 system_prompt: str | None, engine_name: str = "react"):
        self._graph = graph
        self.engine_name = engine_name
        # Keep the original tools dict so dry-run inspection matches AgentLoop.
        self.tools = tools
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        # Conversation memory mirroring Session: run() resends accumulated
        # LangChain messages because the graph has no checkpointer.
        self._messages: list = []
        # langgraph counts graph node visits (~2 visits per tool round plus answer).
        self._recursion_limit = max_iterations * 2 + 1
        # True when the last run() could not answer. The framework engines
        # keep no per-turn flags, so this is how `ovat run` and bench learn
        # that a run failed: the failure sentence below is otherwise just text, and a
        # capped run exited 0 like a good one.
        self.last_failed = False
        # The run trace, in the native loop's shape: what the last run()
        # cost. Empty until a run has finished.
        self.last_trace: dict = {}

    def run(self, user_message: str) -> str:
        """Run the LangChain agent for one message and return the final text."""
        from langgraph.errors import GraphRecursionError

        self.last_failed = False
        started = time.monotonic()
        try:
            result = self._graph.invoke(
                {"messages": [*self._messages, ("user", user_message)]},
                config={"recursion_limit": self._recursion_limit},
            )
        except GraphRecursionError:
            # History is left untouched: a failed run must not poison the next
            # question with a half-finished exchange.
            # Same wording as the native loop so the two engines fail alike.
            self.last_failed = True
            # The graph raised before handing back any messages, so what the
            # run cost is unknown, not zero.
            self.last_trace = framework_trace(self.engine_name, None, None,
                                              True, started)
            return (f"Error: I reached my max of {self.max_iterations} steps "
                    f"without a final answer.")
        # Everything after the history and this question is what the run
        # added: one AIMessage per model call, one ToolMessage per tool that
        # actually ran.
        new = result["messages"][len(self._messages) + 1:]
        self.last_trace = framework_trace(
            self.engine_name, _usage_per_call(new),
            sum(1 for m in new if getattr(m, "type", None) == "tool"),
            False, started)
        # The graph returns the FULL message list (input plus everything new),
        # so storing it back is the whole memory update.
        self._messages = result["messages"]
        final = result["messages"][-1]
        return getattr(final, "content", str(final))


def build_react_agent(config: WorkflowConfig, tools: dict,
                      llm=None) -> LangChainAgent:
    """Build the LangChain react agent. ``llm`` is injectable for testing.

    Production callers build a ``ChatOpenAI`` pointed at OVMS; unit tests inject
    a fake chat model to exercise tool calling without a live server.
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
                          config.agent.system_prompt,
                          engine_name=config.agent.type)
