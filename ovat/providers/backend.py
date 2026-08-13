# ovat/providers/backend.py
"""One description of an OpenAI-compatible backend, read by every engine.

The problem this solves: OVAT has ONE backend (OVMS) and FOUR engines that each
wrap it in their own client class. LangChain wants a ChatOpenAI, LlamaIndex an
OpenAILike, the Agents SDK an OpenAIChatCompletionsModel, the native loop an
OVMSLLMProvider. Each of those four sites separately knew the url, the model
name, the timeout and the sampling temperature.

Four places that must agree, with nothing making them agree, is not a style
problem: `temperature` was set in two of the four and omitted in the other two
for an entire release. The benchmark then compared determinism against
sampling and reported it as a latency difference between frameworks.

So the connection is described ONCE here, and each engine reads it. Adding a
setting (top_p, seed, a header) becomes one edit instead of four, and the four
cannot silently drift apart again.

Deliberately a plain frozen dataclass, not a class with behaviour: it holds no
client, opens no socket, and is safe to build on a machine with no server. The
engines own their own client construction, because each framework's client
takes different arguments; what they share is the ANSWER to "what am I dialling
and how".
"""
from dataclasses import dataclass

from ovat.config.workflow import WorkflowConfig


@dataclass(frozen=True)
class LLMBackend:
    """Everything needed to reach an OpenAI-compatible LLM endpoint.

    frozen=True so an engine cannot mutate the shared description out from
    under the others; anything that needs a variant makes its own copy.
    """

    url: str
    model: str
    temperature: float
    timeout: float
    # The ceiling on ONE reply. Lives here, beside temperature, because all
    # four engines read this description -- and an unbounded generation is not
    # an engine's problem to solve four times.
    max_tokens: int | None = None
    # OVMS ignores the key entirely, but every OpenAI-shaped SDK requires the
    # field to be present, so it is part of the description rather than a
    # literal repeated at four call sites.
    api_key: str = "not-needed"

    @classmethod
    def from_config(cls, config: WorkflowConfig) -> "LLMBackend":
        """The one place the model: section becomes a connection."""
        m = config.model
        return cls(url=m.ovms_url, model=m.name, temperature=m.temperature,
                   timeout=m.request_timeout, max_tokens=m.max_tokens)
