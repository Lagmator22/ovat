# ovat/tui/server/methods/session.py
"""Reading the session, its history, the `!` shell, and small settings."""
from ovat.tui.server import views
from ovat.tui.server.rpc import Reply, Routes, RpcError
from ovat.tui.vibe.app_server.models import PublicHistoryEntry
from ovat.tui.vibe.app_server.protocol import (AgentSwitchParams,
                                               EmptyResponse,
                                               ProtocolErrorCode,
                                               RuntimeMutationResponse,
                                               SessionHistoryClearParams,
                                               SessionHistoryClearResponse,
                                               SessionHistoryListParams,
                                               SessionHistoryListResponse,
                                               SessionListParams,
                                               SessionListResponse,
                                               SessionLogReadParams,
                                               SessionLogReadResponse,
                                               SessionReadParams,
                                               SessionReadResponse,
                                               SessionReadyReadParams,
                                               SessionReadyReadResponse,
                                               SessionReadyWaitParams,
                                               SessionReadyWaitResponse,
                                               SessionSettingsUpdateParams,
                                               SessionShellCommandParams)

routes = Routes()


@routes.method("session/read", SessionReadParams)
async def session_read(ctx, params: SessionReadParams):
    projector = ctx.require_session(params.session_id)
    state = projector.state(
        history_limit=params.history_limit if params.include_history else None,
        turns_limit=params.turns_limit if params.include_turns else None)
    return SessionReadResponse(state=state, last_event_id=state.event_id)


@routes.method("session/list", SessionListParams)
async def session_list(ctx, params: SessionListParams):
    # Nothing is saved, so there is nothing to list and nothing to continue.
    return SessionListResponse()


@routes.method("session/log/read", SessionLogReadParams)
async def session_log_read(ctx, params: SessionLogReadParams):
    projector = ctx.require_session(params.session_id)
    return SessionLogReadResponse(log=views.session_log(projector.session.id))


@routes.method("session/ready/wait", SessionReadyWaitParams)
async def ready_wait(ctx, params: SessionReadyWaitParams):
    ctx.require_session(params.session_id)
    await ctx.wait_ready()
    return SessionReadyWaitResponse(ready=True, init_duration_ms=None)


@routes.method("session/ready/read", SessionReadyReadParams)
async def ready_read(ctx, params: SessionReadyReadParams):
    ctx.require_session(params.session_id)
    return SessionReadyReadResponse(ready=ctx.is_ready)


# ---- history --------------------------------------------------------------------

def history_page(entries: list[PublicHistoryEntry], *, turn_id: str | None,
                 before: str | None, after: str | None, limit: int):
    """(page, cursor_before, cursor_after): the same paging as Vibe's server.
    The cursors are entry ids; `before` is set when older entries remain,
    `after` when newer ones do."""
    if turn_id is not None:
        entries = [e for e in entries if e.turn_id == turn_id]
    if after is not None:
        start = next((i + 1 for i, e in enumerate(entries) if e.id == after),
                     len(entries))
        page = entries[start:start + limit]
    elif before is not None:
        end = next((i for i, e in enumerate(entries) if e.id == before), 0)
        page = entries[max(0, end - limit):end]
    else:
        page = entries[-limit:]
    first = entries.index(page[0]) if page else -1
    last = entries.index(page[-1]) if page else -1
    return (page,
            page[0].id if first > 0 else None,
            page[-1].id if 0 <= last < len(entries) - 1 else None)


@routes.method("session/history/list", SessionHistoryListParams)
async def history_list(ctx, params: SessionHistoryListParams):
    projector = ctx.require_session(params.session_id)
    backward = params.sort_direction == "backward"
    page, older, newer = history_page(
        projector.history, turn_id=params.turn_id,
        before=params.cursor if backward else None,
        after=None if backward else params.cursor, limit=params.limit)
    return SessionHistoryListResponse(
        items=[e.model_copy(deep=True) for e in page],
        next_cursor=older if backward else newer,
        previous_cursor=newer if backward else older)


@routes.method("session/history/clear", SessionHistoryClearParams)
async def history_clear(ctx, params: SessionHistoryClearParams):
    ctx.require_session(params.session_id)
    async with ctx.lifecycle:
        await ctx.new_conversation()
    projector = ctx.projector
    return SessionHistoryClearResponse(
        state=projector.state(),
        session_log=views.session_log(projector.session.id))


# ---- the `!` shell ---------------------------------------------------------------

@routes.method("session/shellCommand", SessionShellCommandParams)
async def shell_command(ctx, params: SessionShellCommandParams):
    return await ctx.shell.run(params)


# ---- settings and the agent profile ----------------------------------------------

@routes.method("session/settings/update", SessionSettingsUpdateParams)
async def settings_update(ctx, params: SessionSettingsUpdateParams):
    ctx.require_session(params.session_id)
    if not ctx.is_ready:
        raise RpcError(ProtocolErrorCode.CONFLICT,
                       "The agent is not ready yet")
    agent = ctx.agent.agent
    if params.max_turns is not None:
        if not hasattr(agent, "max_iterations"):
            raise RpcError(
                ProtocolErrorCode.NOT_IMPLEMENTED,
                "not supported in OVAT: max_turns on this engine "
                "(only the native engine has a step limit)")
        if params.max_turns > 0:
            agent.max_iterations = params.max_turns
    if params.max_tokens is not None:
        llm = getattr(agent, "llm", None)
        if not hasattr(llm, "max_tokens"):
            raise RpcError(
                ProtocolErrorCode.NOT_IMPLEMENTED,
                "not supported in OVAT: max_tokens on this engine "
                "(only the native engine can change it)")
        if params.max_tokens > 0:
            llm.max_tokens = params.max_tokens
    return EmptyResponse()


@routes.method("session/agent/update", AgentSwitchParams)
async def agent_update(ctx, params: AgentSwitchParams):
    ctx.require_session(params.session_id)
    chosen = views.profile(params.agent_name)
    if chosen is None:
        raise RpcError(ProtocolErrorCode.NOT_FOUND,
                       f"No agent profile named {params.agent_name}")
    ctx.profile = chosen.name
    # /yolo in the old UI is this switch: the broker says yes to every ask.
    ctx.agent.broker.auto_approve = chosen.name == views.AUTO_APPROVE_PROFILE
    return Reply(RuntimeMutationResponse(runtime=ctx.runtime()),
                 after=ctx.refresh_runtime)
