# ovat/tui/server/methods/config.py
"""Reading the configuration. Read-only for now: a workflow is edited in its
YAML, and writing it back from the app comes in a later phase."""
from ovat.tui.server import views
from ovat.tui.server.rpc import Routes
from ovat.tui.vibe.app_server.protocol import (ConfigEffectiveReadParams,
                                               ConfigEffectiveReadResponse,
                                               ConfigReadParams,
                                               ConfigReadResponse,
                                               ConfigSchemaReadParams,
                                               ConfigSchemaReadResponse)

routes = Routes()

SCHEMA_VERSION = "ovat-workflow-1"


@routes.method("config/read", ConfigReadParams)
async def config_read(ctx, params: ConfigReadParams):
    if params.session_id is not None:
        ctx.require_session(params.session_id)
    return ConfigReadResponse(
        config=views.config_view(ctx.info, ctx.workflow),
        skills_count=0, hooks_count=0, mcp_servers_total=0,
        mcp_servers_enabled=0)


@routes.method("config/effective/read", ConfigEffectiveReadParams)
async def config_effective_read(ctx, params: ConfigEffectiveReadParams):
    ctx.require_session(params.session_id)
    info = ctx.info
    # Built from what the banner already shows, so nothing secret (an MCP
    # server's env, a token) can end up in it.
    return ConfigEffectiveReadResponse(config={
        "workflow": ctx.workflow,
        "profile": ctx.profile,
        "model": {"name": info.get("model"), "provider": info.get("provider"),
                  "device": info.get("device")},
        "engine": info.get("engine"),
        "tools": [{k: t.get(k) for k in ("name", "kind", "permission")}
                  for t in info.get("tools", [])],
        "subagents": [{k: a.get(k) for k in ("name", "where")}
                      for a in info.get("subagents", [])]})


@routes.method("config/schema", ConfigSchemaReadParams)
async def config_schema(ctx, params: ConfigSchemaReadParams):
    from ovat.config.workflow import WorkflowConfig
    return ConfigSchemaReadResponse(
        config_schema_version=SCHEMA_VERSION,
        config_schema=WorkflowConfig.model_json_schema())
