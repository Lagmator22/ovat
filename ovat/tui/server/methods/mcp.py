"""Vibe's MCP panel over workflow mcp_stdio entries, without cloud auth.

Mirrors Vibe's app_server/_projection.py MCP source pictures. Selections
are held by AgentSession and rebuilt through the normal memory-carrying path.
"""
from ovat.tui.server.rpc import Routes, RpcError
from ovat.tui.server.views import mcp_state
from ovat.tui.vibe.app_server.protocol import (
    ConnectorCatalogReadParams, ConnectorCatalogReadResponse,
    ConnectorCatalogView, MCPCatalogMutationResponse, MCPReadParams,
    MCPReadResponse, MCPRefreshParams, MCPToggleParams, ProtocolErrorCode)

routes = Routes()


@routes.method("mcp_catalog/read", MCPReadParams)
async def read(ctx, params):
    ctx.require_session(params.session_id)
    if hasattr(ctx.agent, "refresh_mcp"):
        ctx.agent.refresh_mcp()
        ctx.info = ctx.agent.info
    return MCPReadResponse(mcp=mcp_state(ctx.info))


@routes.method("connector_catalog/read", ConnectorCatalogReadParams)
async def connectors(ctx, params):
    ctx.require_session(params.session_id)
    return ConnectorCatalogReadResponse(catalog=ConnectorCatalogView(
        disposition="memory"))


@routes.method("mcp_catalog/refresh", MCPRefreshParams)
async def refresh(ctx, params):
    await read(ctx, params)
    return MCPCatalogMutationResponse(runtime=ctx.runtime())


@routes.method("mcp_catalog/toggle", MCPToggleParams)
async def toggle(ctx, params):
    ctx.require_session(params.session_id)
    if ctx.turns.busy or not ctx.is_ready:
        raise RpcError(ProtocolErrorCode.CONFLICT,
                       "Wait for the current turn and build to finish first.")
    source = next((x for x in ctx.info.get("mcp", ())
                   if x["name"] == params.name), None)
    if source is None or params.source != "server":
        raise RpcError(ProtocolErrorCode.NOT_FOUND,
                       "That MCP server is not in this workflow.")
    if params.tool_name and params.tool_name not in {
            t["name"] for t in source["tools"]}:
        raise RpcError(ProtocolErrorCode.NOT_FOUND,
                       "That tool is not offered by this MCP server.")
    selection = ctx.agent.mcp_selection
    old = (set(selection.disabled), set(selection.disabled_tools))
    selection.toggle(params.name, params.disabled, params.tool_name)
    try:
        failure = await ctx.pick()
    except BaseException:
        selection.disabled, selection.disabled_tools = old
        raise
    if failure:
        selection.disabled, selection.disabled_tools = old
        raise RpcError(ProtocolErrorCode.INTERNAL_ERROR, failure)
    return MCPCatalogMutationResponse(runtime=ctx.runtime())
