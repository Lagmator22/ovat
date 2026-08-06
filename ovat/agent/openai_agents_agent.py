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

from ovat.text import strip_code_fence
from ovat.agent.arg_models import json_schema_for_tool
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

    return ModelSettings(temperature=LLMBackend.from_config(config).temperature)


class OpenAIAgentsAgent:
    """Adapter so an Agents SDK agent looks like my native AgentLoop."""

    def __init__(self, agent, tools: dict, max_iterations: int,
                 system_prompt: str | None):
        self._agent = agent
        self.tools = tools
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        # Conversation memory, same contract as the native loop's Session.
        # The SDK is stateless per run; result.to_input_list() returns the
        # full transcript (input plus everything the run added), which is
        # exactly the input the next run should start from.
        self._input_items: list = []

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

        from agents.exceptions import MaxTurnsExceeded

        try:
            result = asyncio.run(Runner.run(
                self._agent,
                [*self._input_items, {"role": "user", "content": user_message}],
                # One "turn" is a model call plus its tool round, which is the
                # same unit the native loop caps, so the number means the
                # same thing in both engines.
                max_turns=self.max_iterations,
            ))
        except MaxTurnsExceeded:
            # History is left untouched: a failed run must not poison the next
            # question with a half-finished exchange.
            # Same wording as the native loop so every engine fails alike.
            return (f"Error: I reached my max of {self.max_iterations} steps "
                    f"without a final answer.")
        self._input_items = result.to_input_list()
        return str(getattr(result, "final_output", result) or "").strip()


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
                             config.agent.system_prompt)
