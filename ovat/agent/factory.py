# ovat/agent/factory.py
"""Layer 2: the component factory, config in, ready-to-run agent out.

Design note: this is where the whole architecture pays off. The factory reads
a validated WorkflowConfig and builds the real objects: the LLM provider, the
tools, and the AgentLoop, all wired together. The user never constructs anything
by hand; they write YAML, the factory does the rest.

This is also where the ABC bet from base.py cashes in. The config says
device: GPU or a model name as a STRING, and the factory picks the matching
provider. Swapping a backend is a one-line YAML edit, not a code change.
"""
from ovat.agent.loop import AgentLoop
from ovat.config.workflow import WorkflowConfig
from ovat.providers.llm_ovms import OVMSLLMProvider
from ovat.tools import search_docs as search_docs_tool
from ovat.tools import transcribe as transcribe_tool


# My registry of built-in tools. Each entry is exactly what the agent loop
# wants: a schema (the menu the model reads) and a function (the real callable).
# I wrap each impl in a small lambda so the loop can call function(**args)
# without knowing about the optional backend argument.
BUILTIN_TOOLS = {
    "search_docs": {
        "schema": search_docs_tool.SCHEMA,
        "function": lambda query, top_k=5: search_docs_tool.search_docs_impl(query, top_k),
    },
    "transcribe": {
        "schema": transcribe_tool.SCHEMA,
        "function": lambda file_path, language="en": transcribe_tool.transcribe_impl(file_path, language),
    },
}


def build_llm(config: WorkflowConfig) -> OVMSLLMProvider:
    """I build the LLM provider from the model section of the config.

    Design note: constructing OVMSLLMProvider does NOT connect to the server,
    it only sets up the client. So this is safe to call without OVMS running,
    which is why the factory tests pass on the Mac.
    """
    m = config.model
    return OVMSLLMProvider(base_url=m.ovms_url, model=m.name)


def build_tools(config: WorkflowConfig) -> dict:
    """I turn the list of tool configs into the dict my loop expects.

    Heads up: only built-in tools are supported right now. If the YAML names a
    tool I do not have, I raise a clear error instead of silently giving the
    agent an empty toolbox.
    """
    tools = {}
    for tool_cfg in config.tools:
        if tool_cfg.type != "builtin":
            raise ValueError(
                f"Unsupported tool type '{tool_cfg.type}' for '{tool_cfg.name}'. "
                f"Only 'builtin' is supported right now."
            )
        if tool_cfg.name not in BUILTIN_TOOLS:
            raise ValueError(
                f"Unknown builtin tool '{tool_cfg.name}'. "
                f"Available: {list(BUILTIN_TOOLS)}"
            )
        tools[tool_cfg.name] = BUILTIN_TOOLS[tool_cfg.name]
    return tools


def build_agent(config: WorkflowConfig) -> AgentLoop:
    """I assemble the full agent from a config: the one call the CLI makes."""
    llm = build_llm(config)
    tools = build_tools(config)
    return AgentLoop(
        llm=llm,
        tools=tools,
        system_prompt=config.agent.system_prompt,
        max_iterations=config.agent.max_iterations,
    )
