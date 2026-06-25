# ovat/config/workflow.py
"""Layer 1: the workflow config, my whole project in one YAML file.

Note: this is the heart of OVAT's promise, "one YAML + one command".
A user writes a small workflow.yml describing which model, which tools, and
how the agent should behave. This file turns that YAML into a VALIDATED Python
object, so the rest of my code never reads raw dicts or worries about typos.

I use pydantic for validation. A pydantic BaseModel is like a struct with a
built-in contract: if the YAML is missing a field or has the wrong type,
pydantic raises a clear error instead of letting a bad value sneak deep into
my code and crash later somewhere confusing.

The YAML I am parsing looks like this:

    model:
      name: Qwen3-8B-int4-ov
      device: GPU
      ovms_url: http://localhost:8000/v3
      tool_parser: hermes3
    tools:
      - name: search_docs
        type: builtin
    agent:
      type: native
      max_iterations: 10
"""
import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Which model to talk to and how. Mirrors the OVMS serving settings."""

    name: str                                   # the model name OVMS serves
    device: str = "CPU"                         # CPU, GPU, or NPU
    ovms_url: str = "http://localhost:8000/v3"  # where my OVMS server listens
    tool_parser: str = "hermes3"                # how to decode tool calls
    # reasoning_parser is for thinking models like the Qwen3 30B variant. It
    # stays None for normal models, which is why I default it to None.
    reasoning_parser: str | None = None


class ToolConfig(BaseModel):
    """One tool the agent is allowed to use."""

    name: str                       # must match a tool I know how to build
    # "builtin" means one of my own tools (search_docs, transcribe). Later I
    # can add "mcp_stdio" to launch an external MCP server as a subprocess.
    type: str = "builtin"
    command: list[str] | None = None  # only used by mcp_stdio launch later


class AgentConfig(BaseModel):
    """How the agent loop behaves."""

    # "native" uses my own loop.py. Later "react" will use LangChain instead.
    type: str = "native"
    max_iterations: int = 10            # the safety cap from my loop
    system_prompt: str | None = None    # optional persona for the agent


class WorkflowConfig(BaseModel):
    """The whole workflow: one model, some tools, one agent."""

    model: ModelConfig
    # default_factory=list gives each config its own empty list. I never share
    # one list between objects, which is a classic mutable-default bug in Python.
    tools: list[ToolConfig] = Field(default_factory=list)
    agent: AgentConfig = Field(default_factory=AgentConfig)


def load_workflow(path: str) -> WorkflowConfig:
    """I read a YAML file from disk and validate it into a WorkflowConfig.

    How it works: yaml.safe_load turns the file into plain dicts and lists.
    Then WorkflowConfig(**data) hands those to pydantic, which checks every
    field and raises a readable error if something is wrong. Two steps: parse
    the text, then validate the shape.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return WorkflowConfig(**data)
