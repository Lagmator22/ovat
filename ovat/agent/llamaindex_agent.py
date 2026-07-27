# ovat/agent/llamaindex_agent.py
"""Layer 3 (alternate engine): run the same agent through LlamaIndex.

The proposal names LlamaIndex as the second framework integration, so this is
the `agent.type: llamaindex` path. As with the LangChain engine, the rest of
OVAT must not care which engine runs: AgentLoop exposes .run(text) -> text, so
does LangChainAgent, and so does LlamaIndexAgent below. The factory hands back
one of the three and the CLI calls .run() either way.

Under the hood this is LlamaIndex's FunctionAgent driven by an OpenAILike LLM
pointed at the OVMS /v3 endpoint. OpenAILike exists precisely for
OpenAI-compatible servers that are not OpenAI, which is exactly what OVMS is.
Two flags on it matter and are easy to get wrong:

  * is_chat_model=True, or LlamaIndex calls the legacy completions endpoint
    that OVMS does not serve for these models.
  * is_function_calling_model=True, or FunctionAgent decides the model cannot
    call tools and silently degrades to plain chat, which looks like the
    tools were never configured.

FunctionAgent is async, and OVAT's .run() is synchronous everywhere else, so
this adapter owns the event loop. It refuses to run inside an existing loop
rather than corrupting one: see run().

llama_index is imported lazily inside the build function so `import ovat` stays
cheap and someone using only the native loop never pays for the install.
"""
import asyncio

from ovat.agent.arg_models import args_model_from_schema
from ovat.config.workflow import WorkflowConfig


def _wrap_tools(tools: dict) -> list:
    """Turn my {name: {schema, function}} dict into LlamaIndex FunctionTools.

    Each wrapped tool reuses the exact same callable the native loop runs, so
    a tool behaves identically no matter which engine called it. The retriever
    bound into search_docs by the factory is already inside that callable.
    """
    from llama_index.core.tools import FunctionTool

    wrapped = []
    for name, spec in tools.items():
        wrapped.append(FunctionTool.from_defaults(
            fn=spec["function"],
            name=name,
            description=spec["schema"]["function"]["description"],
            # Derived from the tool's own SCHEMA, never hand-written: see
            # ovat/agent/arg_models.py for why that rule exists.
            fn_schema=args_model_from_schema(name, spec["schema"]),
        ))
    return wrapped


def _build_llm(config: WorkflowConfig):
    """Build an OpenAILike LLM pointed at OVMS. No network call happens here."""
    from llama_index.llms.openai_like import OpenAILike

    m = config.model
    # api_key is required by the SDK but ignored by OVMS; any string works.
    # temperature 0 keeps demo answers stable across runs, matching the
    # LangChain engine so the two can be compared fairly in a benchmark.
    return OpenAILike(
        model=m.name,
        api_base=m.ovms_url,
        api_key="not-needed",
        temperature=0,
        is_chat_model=True,
        is_function_calling_model=True,
        timeout=m.request_timeout,
    )


class LlamaIndexAgent:
    """Adapter so a LlamaIndex FunctionAgent looks like my native AgentLoop."""

    def __init__(self, agent, tools: dict, max_iterations: int,
                 system_prompt: str | None):
        self._agent = agent
        # The original tools dict is kept so `ovat run --dry-run` can print the
        # tool names the same way it does for the native loop.
        self.tools = tools
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt

    def run(self, user_message: str) -> str:
        """Run the agent for one message and return the final text.

        FunctionAgent is async and everything else in OVAT is not, so the
        event loop is owned here. asyncio.run REFUSES to nest, and that
        refusal is deliberately not worked around: a running loop means we are
        being called from async code (the TUI's worker, say), and quietly
        spawning a second loop there is how you get a deadlock that only
        shows up on someone else's machine.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass                      # no loop running: the normal CLI case
        else:
            raise RuntimeError(
                "The llamaindex engine cannot be run from inside an async "
                "event loop. Call it from a worker thread, or use "
                "agent.type: native.")
        return asyncio.run(self._arun(user_message))

    async def _arun(self, user_message: str) -> str:
        response = await self._agent.run(user_message)
        # FunctionAgent returns a response object whose str() is the answer.
        # Reading .response first keeps the text clean when the object grows
        # extra repr detail, which it has done between releases.
        return str(getattr(response, "response", response) or "").strip()


def build_llamaindex_agent(config: WorkflowConfig, tools: dict,
                           llm=None) -> LlamaIndexAgent:
    """Build the LlamaIndex agent. `llm` is injectable for testing.

    In production this builds an OpenAILike against OVMS. In tests a fake LLM
    is passed so the whole tool-calling loop runs on any machine with no
    server and no network.
    """
    try:
        from llama_index.core.agent.workflow import FunctionAgent
    except ImportError as exc:
        raise RuntimeError(
            "agent.type 'llamaindex' needs LlamaIndex. Install it with: "
            "pip install 'ovat[llamaindex]'"
        ) from exc

    agent = FunctionAgent(
        tools=_wrap_tools(tools),
        llm=llm if llm is not None else _build_llm(config),
        system_prompt=config.agent.system_prompt,
    )
    return LlamaIndexAgent(agent, tools, config.agent.max_iterations,
                           config.agent.system_prompt)
