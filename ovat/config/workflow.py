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
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base for every config section: unknown YAML keys are ERRORS.

    pydantic's default silently IGNORES keys it does not know, so a typo like
    `max_iteration:` (missing s) would just... do nothing, and the default
    would quietly apply. extra="forbid" turns typos into immediate, named
    errors; for a config-driven toolkit that is the whole safety story.
    """

    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    """Which model to talk to and how. Mirrors the OVMS serving settings."""

    name: str                                   # the model name OVMS serves
    # Which LLM backend runs this model. The same knob EmbeddingsConfig and
    # RetrieverConfig already have; ModelConfig was the one place it was
    # missing, which is why build_llm could only ever return OVMS.
    #   ovms  = the OVMS server at ovms_url (tool calling works)
    #   genai = local openvino_genai on this machine, NO server needed, but
    #           it cannot request tools at all (see llm_genai.chat), so it is
    #           plain chat rather than an agent. `ovat chat` is the better
    #           local path; this exists so the contract is honest.
    provider: str = "ovms"
    device: str = "CPU"                         # CPU, GPU, or NPU
    ovms_url: str = "http://localhost:8000/v3"  # where my OVMS server listens
    # Port for OVMS to bind to when running via `ovat serve`
    ovms_port: int = 8000
    # How OVMS decodes tool calls. None means "derive it from the model name"
    # (see _derive_tool_parser below); an explicit value always wins.
    tool_parser: str | None = None
    # reasoning_parser is for thinking models like the Qwen3 30B variant. It
    # stays None for normal models, which is why I default it to None.
    reasoning_parser: str | None = None
    # These two only matter for `ovat serve`, which starts OVMS for me. They
    # tell OVMS where to find (or download) the model. Without them, serve points
    # OVMS at a relative "models" folder with nothing in it, so it cannot start.
    source_model: str | None = None             # HF id, e.g. OpenVINO/Qwen3-8B-int4-ov
    model_repository_path: str = "models"       # folder on disk where models live
    # Cap on ONE HTTP request to OVMS. Without any cap the OpenAI SDK waits
    # ~10 minutes on a hung server, so a number is still needed; the question
    # is only which one.
    #
    # 120s looked generous and was not. Measured on a CPU-only Ubuntu 24.04
    # box with Qwen3.5-0.8B: a plain 32-token completion comes back in ONE
    # SECOND, so the server is plainly healthy -- but an agent request carries
    # the system prompt, the tool schemas and the whole history, and a verbose
    # small model then generates for minutes on CPU. The first cold run took
    # 1056s. What the user saw was "Request timed out." from a server that was
    # answering fine, which reads as a broken install on the very first
    # command a new user runs.
    #
    # 20 minutes. That is long enough that no honest CPU generation is cut
    # off, and the cost is the opposite case: a genuinely wedged server now
    # takes 20 minutes to say so rather than 2. That trade is deliberate. A
    # slow machine is COMMON and the failure is silent and confusing; a hung
    # server is RARE and, when it happens, `ovat doctor` names it in seconds.
    # Anyone who wants the old behaviour sets this to 120 in their workflow.
    request_timeout: float = 1200.0
    # Sampling temperature, applied by EVERY engine. It lives here rather
    # than in each engine because two of them used to hardcode 0 while the
    # other two took the server's default, which made a cross-engine
    # benchmark meaningless: react and llamaindex returned byte-identical
    # answers across repeated runs while native and openai-agents varied, so
    # their tight run-to-run spread measured determinism, not stability.
    # 0.0 by default: an agent that has to emit well-formed tool calls wants
    # the most likely token, and a reproducible run is worth more here than a
    # varied one.
    temperature: float = 0.0
    # Ceiling on ONE reply, sent to OVMS as max_tokens.
    #
    # WHY THERE IS A DEFAULT AT ALL. Nothing capped a generation before this,
    # on any engine: OVMSLLMProvider defaulted max_tokens to None and omitted
    # the key, and no config field existed to set one. A model that never
    # emits a stop token therefore generates until the CLIENT gives up, which
    # is 1200s (request_timeout) for `ovat run` and 600s for a bench worker.
    #
    # That is not hypothetical, and it is not one engine's bug. Measured on
    # this machine: llamaindex ran 13 minutes on one request, and on a
    # separate occasion the native loop ran past 600s -- different engines,
    # different positions in the bench, same unbounded generation. Its
    # signature in ovms.log is the KV cache climbing monotonically for the
    # whole run (0.62 -> 3.6 GB in ~9 minutes, one scheduled request), because
    # every new token needs more cache. The cache growth is the SYMPTOM; the
    # runaway is the cause, which is the opposite of how it was first read.
    #
    # Greedy decoding makes it likelier, and this project asks for greedy
    # decoding: temperature defaults to 0.0 just above, and degenerate
    # repetition is a well-known failure mode of it.
    #
    # 4096 is chosen against measurement, not taste. The longest legitimate
    # answers seen here are 458-929 completion tokens, so this is roughly 4x
    # headroom over anything real, while turning a ten-minute hang into a
    # bounded reply that `finish_reason: "length"` labels honestly. Set it to
    # None to restore the old unbounded behaviour.
    max_tokens: int | None = Field(default=4096, gt=0)
    # Prefix caching reuses KV-cache across turns that share a prefix (the
    # whole conversation history does): a big multi-turn speedup. A knob
    # because not every OVMS build/device supports it; was hardcoded before.
    enable_prefix_caching: bool = True
    # Where the ovms executable lives (file or its folder). On Windows OVMS
    # is usually unzipped somewhere (setupvars.bat setups) and NOT on PATH,
    # so `ovat serve`/`ovat models` accept the location here. Also settable
    # via the OVAT_OVMS environment variable; PATH still works too.
    ovms_binary: str | None = None
    # KV cache OVMS may use, in GB. None means "do not pass the flag", so the
    # server keeps its own default and this changes nothing for anyone who
    # does not set it.
    #
    # SETTING THIS CHANGES THE KIND OF CACHE, not just its size. Unset, OVMS
    # allocates dynamically and the cache GROWS -- measured here, 248.5 MB ->
    # 5.6 GB across one session, sitting at or near 100% of whatever was
    # allocated at the time for 61.6% of all readings. Set, the log says
    # `Cache type: static` and the size stops moving (a flat 999.6 MB at
    # --cache_size 1). Only in that second regime does "98-100%" mean the
    # cache is actually full.
    #
    # Why it is worth having. A static cache that fills has a documented
    # consequence: OVMS preempts requests and recomputes them, and "when
    # preemption is not possible ... the request gets terminated when no more
    # cache can be assigned to it, even before reaching stopping criteria"
    # (docs/llm/reference.md). A generation cut there can halve a
    # `<tool_call>` block, and a parser cannot decode a fragment -- it passes
    # it through as prose and the agent answers having run no tool.
    #
    # The earlier note here claimed that mechanism as the measured cause of a
    # 4/4-vs-17/17 result on a DYNAMIC cache. That inference does not hold:
    # with three fifths of all dynamic readings above 95%, "it failed at
    # 98-100%" is a base rate rather than a finding. The mechanism above is
    # documented; that it caused those particular failures is not established.
    #
    # INT, NOT FLOAT, and that is not a detail. OVMS declares this option as
    # `optional uint64 cache_size` (docs/llm/reference.md), and its parser
    # rejects anything with a decimal point. A float field stringified even a
    # whole number as "1.0", so EVERY value of this setting made the server
    # refuse to start:
    #
    #     error parsing options: Argument '1.0' failed to parse
    #
    # which `ovat serve` surfaced only as "OVMS exited without becoming
    # ready". The setting had therefore never worked once since it was added.
    # Whole GB is also all OVMS can express, so an int is the honest type: a
    # request for 1.5 GB is now a config error naming the constraint, rather
    # than a server that silently will not boot.
    ovms_cache_size_gb: int | None = Field(default=None, gt=0)

    # Which OVMS parser suits which model family. Data, not branching, so a
    # new family is one line. Longest prefix first: "qwen3.5" has to be tested
    # before "qwen3", or every Qwen3.5 model matches the Qwen3 rule.
    _PARSER_BY_FAMILY = (("qwen3.5", "qwen3coder"),
                         ("qwen3_5", "qwen3coder"),
                         ("qwen3", "hermes3"),
                         ("qwen2", "hermes3"))
    _DEFAULT_TOOL_PARSER = "hermes3"

    @model_validator(mode="after")
    def _derive_tool_parser(self):
        """Fill in tool_parser from the model name when it was left out.

        The field used to default to a hardcoded "hermes3", which is WRONG for
        Qwen3.5 -- that family emits <function=..><parameter=..> and hermes3
        expects a JSON body, so nothing decodes and the agent answers fluently
        having called no tool. Every shipped config names qwen3coder, so the
        bad default was latent by convention only: any config a user wrote
        themselves without the field hit it.

        "auto" is not the fix either. Measured on live OVMS: with no
        --tool_parser flag it selected no parser at all and returned tool
        calls as text. So OVAT derives the answer from information it already
        has -- the model name -- and falls back to the historical default for
        families it does not recognise, rather than guessing.

        An explicit value is never touched.
        """
        if self.tool_parser is None:
            name = (self.name or "").lower()
            self.tool_parser = next(
                (parser for family, parser in self._PARSER_BY_FAMILY
                 if family in name),
                self._DEFAULT_TOOL_PARSER)
        return self


class ToolConfig(StrictModel):
    """One tool the agent is allowed to use."""

    name: str                       # must match a tool I know how to build
    # "builtin" = one of my own tools (search_docs, transcribe), in-process.
    # "mcp_stdio" = launch `command` as an MCP server subprocess and import
    # every tool it advertises; ANY MCP server plugs in this way.
    type: str = "builtin"
    command: list[str] | None = None  # the mcp_stdio server launch command


class AgentConfig(StrictModel):
    """How the agent loop behaves."""

    # Which engine runs the loop. "native" is my own loop.py; the rest hand
    # the same job to a framework:
    #   react          LangChain      (pip install 'ovat[langchain]')
    #   llamaindex     LlamaIndex     (pip install 'ovat[llamaindex]')
    #   openai-agents  OpenAI Agents  (pip install 'ovat[openai-agents]')
    # Not a Literal on purpose: an unknown value must fail in the factory,
    # where the message can name every supported engine and the extra that
    # installs it, rather than as a pydantic type error at parse time.
    type: str = "native"
    max_iterations: int = 10            # the safety cap from my loop
    system_prompt: str | None = None    # optional persona for the agent


class EmbeddingsConfig(StrictModel):
    """Which embedder turns text into vectors, and where it runs.

    The whole point of pulling this into config is the ABC swap: change
    `provider` from genai to ovms and a different concrete EmbeddingsProvider
    gets built, with no code edit anywhere else.
    """

    provider: str = "genai"     # genai = local openvino_genai; ovms = server /v3
    # For genai this is a path to an OpenVINO model folder on disk. For ovms it
    # is the served model name. Same field, read differently per provider.
    model: str = "models/bge-small-en-v1.5"
    device: str = "CPU"         # genai only: CPU or NPU on the AI PC
    dim: int = 384              # bge-small emits 384 floats; the table must match


class RetrieverConfig(StrictModel):
    """Which vector store holds the chunks and answers nearest-neighbour search."""

    # sqlite-vec is the only backend wired today. usearch/hnsw can slot in later
    # behind the same RetrieverProvider socket without touching the factory call.
    provider: str = "sqlite-vec"
    # A real file path makes the index survive between `ovat index` and `ovat run`.
    db_path: str = "ovat_index.db"


class ChunkConfig(StrictModel):
    """How `ovat index` slices a document before embedding it."""

    size: int = 512        # characters per chunk; roughly a paragraph
    overlap: int = 64      # characters shared with the next chunk so meaning
    #                        is not cut in half at a boundary


class RagConfig(StrictModel):
    """The retrieval-augmented-generation block that powers search_docs.

    Heads up: this whole section is optional. Leave it out and search_docs runs
    in stub mode (handy for wiring tests). Add it and the factory builds a real
    embedder + retriever and search_docs returns real chunks with citations.
    """

    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)


class WorkflowConfig(StrictModel):
    """The whole workflow: one model, some tools, one agent, optional RAG."""

    model: ModelConfig
    # default_factory=list gives each config its own empty list. I never share
    # one list between objects, which is a classic mutable-default bug in Python.
    tools: list[ToolConfig] = Field(default_factory=list)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    # None means "no RAG configured" -> search_docs stays in stub mode.
    rag: RagConfig | None = None


def load_workflow(path: str) -> WorkflowConfig:
    """I read a YAML file from disk and validate it into a WorkflowConfig.

    How it works: yaml.safe_load turns the file into plain dicts and lists.
    Then WorkflowConfig(**data) hands those to pydantic, which checks every
    field and raises a readable error if something is wrong. Two steps: parse
    the text, then validate the shape.
    """
    with open(path, "r", encoding="utf-8") as f:
        # `or {}` because yaml.safe_load returns None for a file that is empty
        # or contains only comments. WorkflowConfig(**None) then raised
        # "argument after ** must be a mapping, not NoneType" -- a Python
        # internals message where the user needed "your workflow has no model
        # section". An empty mapping lets pydantic say that instead.
        data = yaml.safe_load(f) or {}
    return WorkflowConfig(**data)
