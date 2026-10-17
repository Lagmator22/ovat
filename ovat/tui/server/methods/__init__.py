# ovat/tui/server/methods/__init__.py
"""Every method OVAT's app server answers, grouped by what it is about.

A method that is not registered here gets the "not supported in OVAT"
answer from rpc.py; there is no list of refusals to keep in step.

    lifecycle  start, continue, resume, stop
    session    read, ready, history, the `!` shell, settings, agent profile
    turn       turns, the queue behind them, approval answers
    runtime    runtime, stats, tools, agents, diagnostics
    config     config/read, config/effective/read, config/schema (read-only)
    host       account, identity, provider, workspace, feedback, telemetry
"""
from ovat.tui.server.methods import (config, host, lifecycle, mcp, runtime,
                                     session, turn)
from ovat.tui.server.rpc import Routes


def all_routes() -> Routes:
    routes = Routes()
    for module in (lifecycle, session, turn, runtime, config, host, mcp):
        routes.merge(module.routes)
    return routes
