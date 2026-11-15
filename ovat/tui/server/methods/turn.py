# ovat/tui/server/methods/turn.py
"""Turns, the queue behind them, and the answers to approvals.

`turn/steer` and `session/turn/queue/steer` need the native engine (its loop
reads the new instruction before its next model call); any other engine
answers NOT_STEERABLE.
"""
from ovat.tui.server.rpc import Routes
from ovat.tui.vibe.app_server.protocol import (CallbackResultParams,
                                               CallbackResultResponse,
                                               LoopsClearParams,
                                               LoopsClearResponse,
                                               LoopsCreateParams,
                                               LoopsCreateResponse,
                                               LoopsDeleteParams,
                                               LoopsDeleteResponse,
                                               LoopsListParams,
                                               LoopsListResponse,
                                               TurnEnqueueParams,
                                               TurnInterruptParams,
                                               TurnInterruptResponse,
                                               TurnQueueReadParams,
                                               TurnQueueReadResponse,
                                               TurnQueueRemoveParams,
                                               TurnQueueReplaceParams,
                                               TurnQueueResumeParams,
                                               TurnQueueSteerParams,
                                               TurnStartParams,
                                               TurnSteerParams)

routes = Routes()


@routes.method("turn/start", TurnStartParams)
async def turn_start(ctx, params: TurnStartParams):
    return ctx.turns.start(params)


@routes.method("turn/interrupt", TurnInterruptParams)
async def turn_interrupt(ctx, params: TurnInterruptParams):
    projector = ctx.require_session(params.session_id)
    ctx.turns.interrupt(params.expected_turn_id)
    return TurnInterruptResponse(last_event_id=projector.event_id)


@routes.method("turn/steer", TurnSteerParams)
async def turn_steer(ctx, params: TurnSteerParams):
    return ctx.turns.steer(params)


@routes.method("session/turn/queue/steer", TurnQueueSteerParams)
async def queue_steer(ctx, params: TurnQueueSteerParams):
    return ctx.turns.steer_queued(params)


@routes.method("session/turn/enqueue", TurnEnqueueParams)
async def enqueue(ctx, params: TurnEnqueueParams):
    return ctx.turns.enqueue(params)


@routes.method("session/turn/queue/read", TurnQueueReadParams)
async def queue_read(ctx, params: TurnQueueReadParams):
    ctx.require_session(params.session_id)
    return TurnQueueReadResponse(queue=ctx.turns.queue_state())


@routes.method("session/turn/queue/replace", TurnQueueReplaceParams)
async def queue_replace(ctx, params: TurnQueueReplaceParams):
    return ctx.turns.replace(params.queue_item_id, params.as_enqueue_params())


@routes.method("session/turn/queue/remove", TurnQueueRemoveParams)
async def queue_remove(ctx, params: TurnQueueRemoveParams):
    ctx.require_session(params.session_id)
    return ctx.turns.remove(params.queue_item_id)


@routes.method("session/turn/queue/resume", TurnQueueResumeParams)
async def queue_resume(ctx, params: TurnQueueResumeParams):
    ctx.require_session(params.session_id)
    return ctx.turns.resume()


@routes.method("callback/result", CallbackResultParams)
async def callback_result(ctx, params: CallbackResultParams):
    projector = ctx.require_session(params.session_id)
    # A question (ask_user) and an approval share this one method.
    if ctx.questions.owns(params.callback_id):
        ctx.questions.answer(params)
    else:
        ctx.approvals.answer(params)
    return CallbackResultResponse(last_event_id=projector.event_id)


@routes.method("loops/list", LoopsListParams)
async def loops_list(ctx, params: LoopsListParams):
    ctx.require_session(params.session_id)
    return LoopsListResponse(loops=ctx.turns.list_loops())


@routes.method("loops/create", LoopsCreateParams)
async def loops_create(ctx, params: LoopsCreateParams):
    ctx.require_session(params.session_id)
    return LoopsCreateResponse(
        loop=ctx.turns.create_loop(params.interval, params.prompt))


@routes.method("loops/delete", LoopsDeleteParams)
async def loops_delete(ctx, params: LoopsDeleteParams):
    ctx.require_session(params.session_id)
    return LoopsDeleteResponse(loop=ctx.turns.delete_loop(params.loop_id))


@routes.method("loops/clear", LoopsClearParams)
async def loops_clear(ctx, params: LoopsClearParams):
    ctx.require_session(params.session_id)
    return LoopsClearResponse(count=ctx.turns.clear_loops())
