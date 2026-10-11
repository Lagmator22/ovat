# ovat/agentic/__init__.py
"""The agentic layer: what an interactive agent session needs, with no UI.

Note to myself: the TUI should only DRAW. Everything it shows comes from
here, as plain events, so the same session can later drive another front
end (an editor, a web page) and every rule here is testable without a
terminal. No textual import anywhere in this package; a test enforces it.
Permissions (allow | ask | deny, and the broker that asks a person) live
one layer down in ovat/agent/permissions.py, next to the tool wrapper that
enforces them.

    subagents.py     delegating a task to a local workflow, a cloud API
                     model, or a remote A2A agent
    commands.py      the slash-command registry and its fuzzy matcher
    session.py       AgentSession: build, run turns, cancel, `!` shell
"""
