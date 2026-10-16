# ovat/tui/server/methods/lifecycle.py
"""Opening and closing the session.

Sessions are not saved yet, so `session/resume` and `session/continue` have
nothing to open. They answer with a sentence that says so (the app shows it),
except that resuming the session that is already open just returns it.
"""
from ovat.tui.server.rpc import Reply, Routes, RpcError
from ovat.tui.vibe.app_server.protocol import (ProtocolErrorCode,
                                               SessionContinueParams,
                                               SessionResumeParams,
                                               SessionResumeResponse,
                                               SessionStartParams,
                                               SessionStartResponse,
                                               SessionStopParams,
                                               SessionStopResponse)

routes = Routes()

NOT_SAVED = "OVAT does not keep saved sessions yet."


@routes.method("session/start", SessionStartParams)
async def session_start(ctx, params: SessionStartParams):
    async with ctx.lifecycle:
        state, after = ctx.open_session(params.agent_config)
    return Reply(SessionStartResponse(state=state,
                                      last_event_id=state.event_id),
                 after=after)


@routes.method("session/resume", SessionResumeParams)
async def session_resume(ctx, params: SessionResumeParams):
    if ctx.projector is None or ctx.projector.session.id != params.session_id:
        raise RpcError(ProtocolErrorCode.NOT_FOUND,
                       f"Session not found: {params.session_id}. {NOT_SAVED}")
    state = ctx.projector.state(history_limit=params.history_limit)
    return SessionResumeResponse(state=state, last_event_id=state.event_id)


@routes.method("session/continue", SessionContinueParams)
async def session_continue(ctx, params: SessionContinueParams):
    raise RpcError(ProtocolErrorCode.NOT_FOUND,
                   f"There is no saved session to continue. {NOT_SAVED}")


@routes.method("session/stop", SessionStopParams)
async def session_stop(ctx, params: SessionStopParams):
    ctx.require_session(params.session_id)
    return Reply(SessionStopResponse(closed=True), after=ctx.rpc.stop)
