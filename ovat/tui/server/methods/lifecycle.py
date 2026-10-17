# ovat/tui/server/methods/lifecycle.py
"""Opening and closing the session.

`session/resume` opens a saved conversation (by id, or the start of one:
the app prints short ids), `session/continue` the most recent one. With a
session already open they switch to it; the open one stays saved.
"""
from ovat.tui.server import saved
from ovat.tui.server.rpc import Reply, Routes, RpcError
from ovat.tui.vibe.app_server.protocol import (ProtocolErrorCode,
                                               SessionContinueParams,
                                               SessionContinueResponse,
                                               SessionResumeParams,
                                               SessionResumeResponse,
                                               SessionStartParams,
                                               SessionStartResponse,
                                               SessionStopParams,
                                               SessionStopResponse)

routes = Routes()


@routes.method("session/start", SessionStartParams)
async def session_start(ctx, params: SessionStartParams):
    async with ctx.lifecycle:
        state, after = ctx.open_session(params.agent_config)
    return Reply(SessionStartResponse(state=state,
                                      last_event_id=state.event_id),
                 after=after)


@routes.method("session/resume", SessionResumeParams)
async def session_resume(ctx, params: SessionResumeParams):
    if ctx.projector is not None and \
            ctx.projector.session.id == params.session_id:
        state = ctx.projector.state(history_limit=params.history_limit)
        return SessionResumeResponse(state=state, last_event_id=state.event_id)
    if ctx.projector is None:
        ctx.project_dir(params.agent_config)
    rec = saved.find(ctx, params.session_id)
    if rec is None:
        raise RpcError(ProtocolErrorCode.NOT_FOUND,
                       f"No saved session {params.session_id} here "
                       f"(sessions are saved per folder).")
    return await _open(ctx, params, rec, SessionResumeResponse)


@routes.method("session/continue", SessionContinueParams)
async def session_continue(ctx, params: SessionContinueParams):
    if ctx.projector is None:
        ctx.project_dir(params.agent_config)
    store = ctx.store
    records = store.list() if store is not None else []
    if not records:
        raise RpcError(ProtocolErrorCode.NOT_FOUND,
                       "There is no saved session to continue in this "
                       "folder.")
    return await _open(ctx, params, records[0], SessionContinueResponse)


async def _open(ctx, params, rec: dict, response_type):
    async with ctx.lifecycle:
        if ctx.projector is None:
            state, after = ctx.open_session(params.agent_config, restored=rec)
            return Reply(response_type(state=state,
                                       last_event_id=state.event_id),
                         after=after)
        await ctx.switch_to(rec)
    state = ctx.projector.state(history_limit=params.history_limit)
    return response_type(state=state, last_event_id=state.event_id)


@routes.method("session/stop", SessionStopParams)
async def session_stop(ctx, params: SessionStopParams):
    ctx.require_session(params.session_id)
    return Reply(SessionStopResponse(closed=True), after=ctx.rpc.stop)
