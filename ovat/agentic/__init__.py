# ovat/agentic/__init__.py
"""Headless interactive session orchestration decoupled from any terminal UI.

Emits structured event dataclasses to a UI/test sink so front ends and tests
can drive interactive sessions without importing ``textual`` (enforced by
``tests/test_tui_isolation.py``). Tool permission enforcement and brokers live
in ``ovat/agent/permissions.py``.

    subagents.py     task delegation to local workflows, APIs, or A2A peers
    commands.py      slash-command registry and matching
    session.py       AgentSession lifecycle, worker turns, cancellation, `!` shell
"""
