# ovat/tools/delegate.py
"""`delegate`: hand a whole task to a subagent and read its answer.

One tool for every subagent rather than one tool per subagent: the menu the
model reads on every turn stays short, which matters for 4B to 9B models,
and the choice between specialists becomes one argument with a fixed list
of values (a JSON-schema enum). The same shape as OpenCode's `task` tool.

The factory fills the description and the enum from `subagents:` in the
workflow; the real work is ovat/agent/subagents.SubagentRunner.
"""
import copy

SCHEMA = {
    "type": "function",
    "function": {
        "name": "delegate",
        "description": "Hand a self-contained task to a specialist subagent "
                       "and get its answer back. Filled in per workflow.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Which subagent does the task.",
                },
                "task": {
                    "type": "string",
                    "description": "The whole task in plain words. The "
                                   "subagent cannot see this chat, so "
                                   "include everything it needs.",
                },
            },
            "required": ["agent", "task"],
        },
    },
}


def schema_for(subagents) -> dict:
    """SCHEMA with this workflow's subagents named and described."""
    schema = copy.deepcopy(SCHEMA)
    names = [s.name for s in subagents]
    lines = "; ".join(f"{s.name}: {s.description}" for s in subagents)
    schema["function"]["description"] = (
        "Hand a self-contained task to a specialist subagent and get its "
        f"answer back. Subagents: {lines}.")[:900]
    schema["function"]["parameters"]["properties"]["agent"]["enum"] = names
    return schema
