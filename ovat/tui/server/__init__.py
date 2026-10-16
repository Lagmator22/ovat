# ovat/tui/server/__init__.py
"""OVAT's app server: the other end of the copied TUI's wire.

The copied Vibe TUI (ovat/tui/vibe/) never touches an agent. It talks
JSON-RPC to an "app server" through ovat/tui/vibe/app_server/session.py. This
package is that server, written for OVAT's own AgentSession
(ovat/agentic/session.py), so the TUI's code stays an unchanged copy:

    rpc.py         the JSON-RPC loop: framing, dispatch, errors, callbacks
    context.py     one conversation: the AgentSession and what shares it
    projector.py   agent reports -> the session the TUI reads, numbered events
    turns.py       a turn, the queue behind it, how it ends
    approvals.py   the ApprovalBroker as Vibe callbacks
    shell_runs.py  `!command`
    views.py       runtime, config and stats pictures
    methods/       one handler per JSON-RPC method
    bridge.py      tool, subagent and remote events -> history entries

Nothing here imports textual, and nothing talks to a network.

`start_ovat_app_server` wires it all in one process: a memory transport, the
server task, the copied client and its session.
"""
