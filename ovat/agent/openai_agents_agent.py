# ovat/agent/openai_agents_agent.py
"""Layer 3 (alternate engine): run the same agent through the OpenAI Agents SDK.

The proposal names this as the third framework integration, so this is the
`agent.type: openai-agents` path. Same contract as every other engine:
.run(text) -> text, so the factory can hand back any of them and the CLI never
knows the difference.

The SDK is built for api.openai.com, so pointing it at OVMS takes three
deliberate steps, and skipping any one of them fails in a way that looks like
something else:

  * An AsyncOpenAI client with base_url set to the OVMS /v3 endpoint. The SDK
    otherwise reads OPENAI_API_KEY from the environment and talks to OpenAI,
    which on a developer machine with that variable set means your local
    workflow quietly bills a cloud account.
  * OpenAIChatCompletionsModel, NOT the SDK's default Responses model. OVMS
    speaks the chat-completions dialect; the Responses API is OpenAI-only, and
    the failure is a 404 on a path OVMS never claimed to serve.
  * Tracing disabled. Tracing uploads run data to OpenAI's servers and needs a
    real OpenAI key, so leaving it on turns a fully local, private run into a
    network call, which is the exact opposite of the point of running OVMS on
    your own hardware.

Tools are built as explicit FunctionTool objects rather than with the SDK's
@function_tool decorator: the decorator infers a schema from a Python
signature, and OVAT's tools already carry a hand-written SCHEMA that is the
contract. Inferring a second one is how the two drift apart.

`agents` is imported lazily so `import ovat` stays cheap.
"""
import asyncio
import inspect
import json
import time

from ovat.text import strip_code_fence
from ovat.agent.arg_models import json_schema_for_tool
from ovat.agent.usage import framework_trace
from ovat.providers.backend import LLMBackend
from ovat.config.workflow import WorkflowConfig


def _wrap_tools(tools: dict) -> list:
    """Turn my {name: {schema, function}} dict into SDK FunctionTools."""
    from agents import FunctionTool

    def make(name: str, spec: dict):
        function = spec["function"]

        async def on_invoke(context, arguments: str):
            """The SDK hands arguments over as a JSON STRING, not a dict."""
            # Same unwrapping the native loop does. This engine is the only
            # OTHER one that parses arguments itself, so it is the only other
            # one that can be tripped by a fenced payload.
            arguments = strip_code_fence(arguments)
            try:
                kwargs = json.loads(arguments) if arguments else {}
            except (ValueError, TypeError) as exc:
                # Readable string, not an exception: the model is the one that
                # produced this and it is the one that has to recover from it.
                return f"Error: could not parse tool arguments: {exc}"
            try:
                result = function(**kwargs)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:
                return f"Error running {name}: {exc}"
            return result if isinstance(result, str) else json.dumps(result)

        return FunctionTool(
            name=name,
            description=spec["schema"]["function"]["description"],
            params_json_schema=json_schema_for_tool(spec["schema"]),
            on_invoke_tool=on_invoke,
            # OVAT's schemas are hand-written for OVMS and do not all set
            # additionalProperties: false, which strict mode demands. Relaxing
            # it keeps every existing tool usable rather than silently
            # dropping the ones that do not comply.
            strict_json_schema=False,
        )

    return [make(name, spec) for name, spec in tools.items()]


def _build_model(config: WorkflowConfig):
    """An SDK model object bound to OVMS. No network call happens here."""
    from agents import OpenAIChatCompletionsModel
    from openai import AsyncOpenAI

    # One shared description of the connection, so this engine cannot drift
    # from the other three. See ovat/providers/backend.py.
    b = LLMBackend.from_config(config)
    client = AsyncOpenAI(base_url=b.url, api_key=b.api_key, timeout=b.timeout)
    return OpenAIChatCompletionsModel(model=b.model, openai_client=client)


def _model_settings(config: WorkflowConfig):
    """Sampling settings, from the config like every other engine.

    This engine sent none at all, so it took the server's default while two
    others hardcoded 0. That is what made the benchmark's latency column
    compare determinism against sampling.
    """
    from agents import ModelSettings

    b = LLMBackend.from_config(config)
    # ModelSettings has no `seed` field, so it travels in extra_body with the
    # OVMS-only settings; OVMS reads it from the body either way.
    kwargs = b.openai_kwargs()
    extra = b.extra_body()
    if "seed" in kwargs:
        extra["seed"] = kwargs.pop("seed")
    return ModelSettings(temperature=b.temperature, max_tokens=b.max_tokens,
                         extra_body=extra or None, **kwargs)


def _trace_parts(run) -> tuple[list, int]:
    """(per-call usage, tools run) from a RunResult or a RunErrorDetails.

    Both carry raw_responses (one ModelResponse per model call) and
    new_items (one ToolCallOutputItem per tool that actually ran).

    The SDK's own running total, context_wrapper.usage, cannot tell
    "no usage" from zero: when a reply has no usage block it records
    Usage(requests=1) with every count 0 (openai_chatcompletions.py). So each
    reply is read on its own, and one reporting no tokens at all counts as
    unreported. A real reply always has prompt tokens, so this misreads
    nothing OVMS sends.
    """
    from agents.items import ToolCallOutputItem

    calls = []
    for response in run.raw_responses:
        usage = response.usage
        if usage is None or not (usage.input_tokens or usage.output_tokens):
            calls.append((None, None))
        else:
            calls.append((usage.input_tokens, usage.output_tokens))
    ran = sum(isinstance(item, ToolCallOutputItem) for item in run.new_items)
    return calls, ran


class OpenAIAgentsAgent:
    """Adapter so an Agents SDK agent looks like my native AgentLoop."""

    def __init__(self, agent, tools: dict, max_iterations: int,
                 system_prompt: str | None,
                 engine_name: str = "openai-agents"):
        self._agent = agent
        self.engine_name = engine_name
        self.tools = tools
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        # Conversation memory, same contract as the native loop's Session.
        # The SDK is stateless per run; result.to_input_list() returns the
        # full transcript (input plus everything the run added), which is
        # exactly the input the next run should start from.
        self._input_items: list = []
        # True when the last run() could not answer; see LangChainAgent.
        self.last_failed = False
        # What the last run() cost, in the native loop's shape.
        self.last_trace: dict = {}

    def run(self, user_message: str) -> str:
        """Run for one message and return the final text.

        Runner is async and the rest of OVAT is not, so the loop is owned
        here. As in the LlamaIndex engine, running inside an existing loop is
        refused rather than worked around.
        """
        from agents import Runner

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass                      # no loop running: the normal CLI case
        else:
            raise RuntimeError(
                "The openai-agents engine cannot be run from inside an async "
                "event loop. Call it from a worker thread, or use "
                "agent.type: native.")

        from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError

        self.last_failed = False
        started = time.monotonic()
        try:
            result = asyncio.run(Runner.run(
                self._agent,
                [*self._input_items, {"role": "user", "content": user_message}],
                # One "turn" is a model call plus its tool round, which is the
                # same unit the native loop caps, so the number means the
                # same thing in both engines.
                max_turns=self.max_iterations,
            ))
        except MaxTurnsExceeded as exc:
            # History is left untouched: a failed run must not poison the next
            # question with a half-finished exchange.
            # Same wording as the native loop so every engine fails alike.
            self.last_failed = True
            self._record(exc, started)
            return (f"Error: I reached my max of {self.max_iterations} steps "
                    f"without a final answer.")
        except ModelBehaviorError as exc:
            # The MODEL broke the contract, typically by calling a tool that
            # does not exist ("search" for "search_docs"). The SDK ends the
            # run there, and its maintainers declined in-SDK recovery
            # (openai-agents-python#2957). Uncaught, `ovat run` called this
            # "Error talking to OVMS" while the server was fine.
            self.last_failed = True
            self._record(exc, started)
            available = ", ".join(self.tools) or "none"
            return (f"Error: the model broke the tool-calling contract and "
                    f"the run stopped: {exc}. Tools available: {available}. "
                    f"Naming them in agent.system_prompt usually helps; the "
                    f"native engine also recovers near-miss names itself.")
        self._record(result, started)
        self._input_items = result.to_input_list()
        return str(getattr(result, "final_output", result) or "").strip()

    def _record(self, outcome, started: float) -> None:
        """Fill last_trace from a RunResult, or from a failure's run_data.

        On a failure the SDK attaches what the run did before it stopped as
        the exception's run_data, so a failed run still says what it cost.
        When that is missing, the cost is unknown, not zero.
        """
        if isinstance(outcome, Exception):
            outcome = getattr(outcome, "run_data", None)
        calls, ran = (_trace_parts(outcome) if outcome is not None
                      else (None, None))
        self.last_trace = framework_trace(self.engine_name, calls, ran,
                                          self.last_failed, started)


def build_openai_agents_agent(config: WorkflowConfig, tools: dict,
                              model=None) -> OpenAIAgentsAgent:
    """Build the Agents SDK agent. `model` is injectable for testing."""
    try:
        from agents import Agent, set_tracing_disabled
    except ImportError as exc:
        raise RuntimeError(
            "agent.type 'openai-agents' needs the OpenAI Agents SDK. Install "
            "it with: pip install 'ovat[openai-agents]'"
        ) from exc

    # See the module docstring: tracing would upload run data to OpenAI and
    # demand a real key, turning a local private run into a network call.
    set_tracing_disabled(True)

    agent = Agent(
        name="ovat",
        instructions=config.agent.system_prompt,
        model=model if model is not None else _build_model(config),
        model_settings=_model_settings(config),
        tools=_wrap_tools(tools),
    )
    return OpenAIAgentsAgent(agent, tools, config.agent.max_iterations,
                             config.agent.system_prompt,
                             engine_name=config.agent.type)
