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
import time

from ovat.agent.arg_models import args_model_from_schema
from ovat.agent.usage import framework_trace
from ovat.providers.backend import LLMBackend
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

    # One shared description of the connection, so this engine cannot drift
    # from the other three. See ovat/providers/backend.py.
    b = LLMBackend.from_config(config)
    # additional_kwargs reach the OpenAI client's create() call unchanged,
    # so the OVMS-only settings ride along as its extra_body.
    additional = b.openai_kwargs()
    if b.extra_body():
        additional["extra_body"] = b.extra_body()
    return OpenAILike(
        additional_kwargs=additional,
        model=b.model,
        api_base=b.url,
        api_key=b.api_key,
        temperature=b.temperature,
        is_chat_model=True,
        is_function_calling_model=True,
        timeout=b.timeout,
        max_tokens=b.max_tokens,
    )


def _usage_of(raw) -> tuple:
    """(prompt, completion) tokens from one AgentOutput's `raw` reply.

    FunctionAgent hands the OpenAI reply over as a dict (model_dump) on the
    normal path and as the reply object on its early-stopping path, so both
    are read. A reply without a usage block gives (None, None).
    """
    usage = (raw.get("usage") if isinstance(raw, dict)
             else getattr(raw, "usage", None))
    if usage is None:
        return None, None
    if not isinstance(usage, dict):
        usage = {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                 "completion_tokens": getattr(usage, "completion_tokens",
                                              None)}
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


class LlamaIndexAgent:
    """Adapter so a LlamaIndex FunctionAgent looks like my native AgentLoop."""

    def __init__(self, agent, tools: dict, max_iterations: int,
                 system_prompt: str | None, engine_name: str = "llamaindex"):
        self._agent = agent
        self.engine_name = engine_name
        # The original tools dict is kept so `ovat run --dry-run` can print the
        # tool names the same way it does for the native loop.
        self.tools = tools
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        # Conversation memory, same contract as the native loop's Session.
        # Each workflow run is stateless on its own, so the adapter keeps the
        # history and hands it back as chat_history on every call.
        self._history: list = []
        # True when the last run() could not answer; see LangChainAgent.
        self.last_failed = False
        # What the last run() cost, in the native loop's shape. _arun fills
        # the counts below as the workflow's events arrive, so a run that
        # fails part way still reports the calls it made.
        self.last_trace: dict = {}
        self._calls: list = []
        self._tools_ran = 0

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
        from llama_index.core.workflow.errors import WorkflowRuntimeError

        self.last_failed = False
        self._calls, self._tools_ran = [], 0
        started = time.monotonic()
        try:
            return asyncio.run(self._arun(user_message))
        except WorkflowRuntimeError as exc:
            # LlamaIndex's own cap, reached. Only THAT error becomes the
            # native loop's sentence; any other workflow failure is a real
            # error and stays one. Uncaught, `ovat run` reported this as
            # "Error talking to OVMS", blaming a server that was fine.
            if not str(exc).startswith("Max iterations"):
                raise
            self.last_failed = True
            return (f"Error: I reached my max of {self.max_iterations} steps "
                    f"without a final answer.")
        finally:
            self.last_trace = framework_trace(
                self.engine_name, self._calls, self._tools_ran,
                self.last_failed, started)

    def _remember(self, user_message: str, answer: str) -> None:
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        self._history.append(ChatMessage(role=MessageRole.USER,
                                         content=user_message))
        self._history.append(ChatMessage(role=MessageRole.ASSISTANT,
                                         content=answer))

    async def _arun(self, user_message: str) -> str:
        # list() hands the workflow a COPY: it may append to whatever list it
        # is given, and this adapter owns the canonical history.
        #
        # max_iterations is passed, not left to LlamaIndex's default of 20:
        # agent.max_iterations was otherwise ignored on this engine alone.
        handler = self._agent.run(user_message,
                                  chat_history=list(self._history),
                                  max_iterations=self.max_iterations)
        # The workflow's event stream is where the usage is: one AgentOutput
        # per model call (its `raw` is the OpenAI reply, usage included) and
        # one ToolCallResult per tool that actually ran. A test double that
        # returns a plain coroutine has no stream, and records nothing.
        if hasattr(handler, "stream_events"):
            from llama_index.core.agent.workflow import (AgentOutput,
                                                         ToolCallResult)

            async for event in handler.stream_events():
                if isinstance(event, AgentOutput):
                    self._calls.append(_usage_of(event.raw))
                elif isinstance(event, ToolCallResult):
                    self._tools_ran += 1
        response = await handler
        # FunctionAgent returns a response object whose str() is the answer.
        # Reading .response first keeps the text clean when the object grows
        # extra repr detail, which it has done between releases.
        text = str(getattr(response, "response", response) or "").strip()
        # ChatMessage stringifies as "<role>: <content>", so every answer from
        # this engine arrived prefixed with "assistant: ". Harmless to a
        # reader, but it is not part of the model's answer, and it made
        # llamaindex rows in the benchmark differ from the other three
        # engines' by a constant that has nothing to do with the model.
        for role in ("assistant:", "Assistant:"):
            if text.startswith(role):
                text = text[len(role):].lstrip()
                break
        # Recorded only on success: a failed run must not poison the next
        # question with a half-finished exchange.
        self._remember(user_message, text)
        return text


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
        # One plain request per model call, like the other three engines.
        # Streamed, a reply carries usage only when stream_options asks for
        # it, and OVMS documents that option for continuous-batching
        # servables only and usage as not working for streaming on stateful
        # ones (the NPU). Unstreamed, every reply carries it. Nothing here
        # reads the stream token by token: run() waits for the whole answer.
        streaming=False,
    )
    return LlamaIndexAgent(agent, tools, config.agent.max_iterations,
                           config.agent.system_prompt,
                           engine_name=config.agent.type)
