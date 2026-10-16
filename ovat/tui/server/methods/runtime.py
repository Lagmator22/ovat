# ovat/tui/server/methods/runtime.py
"""What the session is running with: runtime, stats, tools, agents."""
from ovat.tui.server import views
from ovat.tui.server.rpc import Routes
from ovat.tui.vibe.app_server.protocol import (AgentsListParams,
                                               AgentsListResponse,
                                               DiagnosticsListParams,
                                               DiagnosticsListResponse,
                                               RuntimeReadParams,
                                               RuntimeReadResponse,
                                               StatsReadParams,
                                               StatsReadResponse,
                                               ToolsListParams,
                                               ToolsListResponse)

routes = Routes()


@routes.method("runtime/read", RuntimeReadParams)
async def runtime_read(ctx, params: RuntimeReadParams):
    projector = ctx.require_session(params.session_id)
    return RuntimeReadResponse(
        runtime=ctx.runtime(),
        session_log=views.session_log(projector.session.id, ctx.store),
        ready=ctx.is_ready)


@routes.method("stats/read", StatsReadParams)
async def stats_read(ctx, params: StatsReadParams):
    ctx.require_session(params.session_id)
    return StatsReadResponse(
        stats=ctx.stats.snapshot(),
        context_window=int(ctx.info.get("context_window") or 0))


@routes.method("tools/list", ToolsListParams)
async def tools_list(ctx, params: ToolsListParams):
    ctx.require_session(params.session_id)
    return ToolsListResponse(tools=views.tool_summaries(ctx.info))


@routes.method("agents/list", AgentsListParams)
async def agents_list(ctx, params: AgentsListParams):
    if params.session_id is not None:
        ctx.require_session(params.session_id)
    active, agents = views.agent_summaries(ctx.info, ctx.profile)
    return AgentsListResponse(active=active, agents=agents)


@routes.method("diagnostics/list", DiagnosticsListParams)
async def diagnostics_list(ctx, params: DiagnosticsListParams):
    ctx.require_session(params.session_id)
    return DiagnosticsListResponse(issues=[], hooks_count=0)
