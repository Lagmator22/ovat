# ovat/tui/server/methods/session.py
"""Reading the session, its history, saved conversations (list, rename,
delete, rewind, fork), the `!` shell, and small settings."""
import asyncio
from uuid import uuid4

from ovat.agent.compaction import CompactionFailed
from ovat.tui.server import saved, views
from ovat.tui.server.projector import PREVIEW_CHARS, now_ms
from ovat.tui.server.rpc import Reply, Routes, RpcError
from ovat.tui.server.turns import BuildFailedError
from ovat.tui.vibe.app_server.models import (PublicCheckpointEntry,
                                             PublicEntryGenerationStatus,
                                             PublicHistoryEntry,
                                             PublicMessageEntry,
                                             PublicSessionState,
                                             PublicTurnQueue)
from ovat.tui.vibe.app_server.protocol import (AgentSwitchParams,
                                               EmptyResponse,
                                               ProtocolErrorCode,
                                               RuntimeMutationResponse,
                                               SessionCompactParams,
                                               SessionCompactResponse,
                                               SessionDeleteParams,
                                               SessionForkParams,
                                               SessionForkResponse,
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
                                               SessionRewindParams,
                                               SessionRewindReadParams,
                                               SessionRewindReadResponse,
                                               SessionRewindResponse,
                                               SessionSettingsUpdateParams,
                                               SessionShellCommandParams,
                                               SessionTitleUpdateParams,
                                               SessionTitleUpdateResponse)

routes = Routes()


@routes.method("session/read", SessionReadParams)
async def session_read(ctx, params: SessionReadParams):
    history_limit = params.history_limit if params.include_history else None
    turns_limit = params.turns_limit if params.include_turns else None
    if ctx.projector is None or ctx.projector.session.id == params.session_id:
        projector = ctx.require_session(params.session_id)
        state = projector.state(history_limit=history_limit,
                                turns_limit=turns_limit)
        return SessionReadResponse(state=state, last_event_id=state.event_id)
    # Another, saved conversation: the /resume picker's preview.
    rec = ctx.store.load(params.session_id) if ctx.store else None
    if rec is None:
        raise RpcError(ProtocolErrorCode.NOT_FOUND,
                       f"Session not found: {params.session_id}")
    return SessionReadResponse(
        state=_saved_state(ctx, rec, history_limit, turns_limit),
        last_event_id=0)


def _saved_state(ctx, rec: dict, history_limit: int | None,
                 turns_limit: int | None) -> PublicSessionState:
    """A saved conversation that is not the open one, as a state."""
    history = saved.history_of(rec, rec["id"])
    turns = saved.turns_of(rec, rec["id"])
    return PublicSessionState(
        event_id=0, session=saved.public_session(rec, ctx),
        history=None if history_limit is None else history[-history_limit:],
        turns=None if turns_limit is None else turns[-turns_limit:],
        active_callbacks=[], turn_queue=PublicTurnQueue())


@routes.method("session/list", SessionListParams)
async def session_list(ctx, params: SessionListParams):
    """The saved conversations of this folder, newest first. The open one is
    among them once it has a turn (it is saved after each)."""
    store = ctx.store
    items = [saved.public_session(r, ctx)
             for r in (store.list() if store is not None else [])]
    return SessionListResponse(
        items=items, continue_session_id=items[0].id if items else None)


@routes.method("session/rename", SessionTitleUpdateParams)
async def session_rename(ctx, params: SessionTitleUpdateParams):
    title = params.title.strip()
    projector = ctx.projector
    if projector is not None and projector.session.id == params.session_id:
        projector.change_session(title=title)
        saved.save(ctx)
        return SessionTitleUpdateResponse(
            title=title, updated_at=str(projector.session.updated_at),
            last_event_id=projector.event_id)
    store = ctx.store
    rec = store.load(params.session_id) if store is not None else None
    if rec is None:
        raise RpcError(ProtocolErrorCode.NOT_FOUND,
                       f"Session not found: {params.session_id}")
    rec.update(title=title, updated_at=now_ms())
    store.save(rec)
    return SessionTitleUpdateResponse(title=title,
                                      updated_at=str(rec["updated_at"]))


@routes.method("session/delete", SessionDeleteParams)
async def session_delete(ctx, params: SessionDeleteParams):
    store = ctx.store
    if store is None or not store.delete(params.session_id):
        raise RpcError(ProtocolErrorCode.NOT_FOUND,
                       f"Session not found: {params.session_id}")
    return EmptyResponse()


@routes.method("session/rewind/read", SessionRewindReadParams)
async def rewind_read(ctx, params: SessionRewindReadParams):
    ctx.require_session(params.session_id)
    # OVAT does not track what tools or `!` commands changed on disk, so a
    # rewind has no files to put back (and the app does not offer to).
    return SessionRewindReadResponse(has_file_changes=False)


@routes.method("session/rewind", SessionRewindParams)
async def session_rewind(ctx, params: SessionRewindParams):
    """Go back to before one of the person's messages: the transcript from
    it on, and what the model remembers from its turn on, are dropped; the
    message itself goes back into the prompt. `inplace` False keeps the
    conversation as it was and continues in a copy (a fork)."""
    projector = _idle_session(ctx, params.session_id, "rewinding")
    history = projector.history
    index = next((i for i, e in enumerate(history)
                  if e.id == params.entry_id), None)
    entry = history[index] if index is not None else None
    if not isinstance(entry, PublicMessageEntry) or entry.role != "user":
        raise RpcError(ProtocolErrorCode.INVALID_PARAMS,
                       "Rewind goes back to one of your messages.")
    if ctx.agent.memory() is None:
        raise RpcError(ProtocolErrorCode.NOT_IMPLEMENTED, NATIVE_ONLY)
    if index < _last_compaction(history):
        raise RpcError(ProtocolErrorCode.INVALID_PARAMS,
                       "That message is from before the conversation was "
                       "compacted, and the model no longer remembers it. "
                       "Pick a message after the compaction.")
    # The first agent turn at or after the entry is where the memory is cut;
    # none (only `@name` questions since) leaves the memory as it is.
    cut = next((ctx.anchors[e.id] for e in history[index:]
                if e.id in ctx.anchors), None)
    if not params.inplace:
        saved.save(ctx)                 # the conversation as it was stays
    if cut is not None:
        try:
            ctx.agent.truncate_memory(cut)
        except ValueError as exc:
            raise RpcError(ProtocolErrorCode.INTERNAL_ERROR,
                           f"Could not rewind the model's memory: {exc}")
    kept = history[:index]
    kept_ids = {e.id for e in kept}
    ctx.anchors = {k: v for k, v in ctx.anchors.items() if k in kept_ids}
    session = projector.session.model_copy(update={
        "preview": next((e.text for e in kept
                         if isinstance(e, PublicMessageEntry)
                         and e.role == "user"), "")[:PREVIEW_CHARS],
        "updated_at": now_ms()})
    if not params.inplace:
        session = session.model_copy(update={"id": str(uuid4()),
                                             "created_at": now_ms()})
    projector.adopt(session, _owned(kept, session.id),
                    _owned([t for t in projector.turns
                            if t.input_entry_id in kept_ids], session.id))
    saved.save(ctx)
    return SessionRewindResponse(
        message=entry.text, restore_errors=[], restored_paths=[],
        state=projector.state(),
        session_log=views.session_log(session.id, ctx.store))


def _last_compaction(history: list[PublicHistoryEntry]) -> int:
    """Index of the newest compaction checkpoint, -1 when there is none."""
    return next((i for i in range(len(history) - 1, -1, -1)
                 if isinstance(history[i], PublicCheckpointEntry)
                 and history[i].kind == "compaction"), -1)


@routes.method("session/compact", SessionCompactParams)
async def session_compact(ctx, params: SessionCompactParams):
    """/compact: the model's memory becomes a summary of itself. The
    transcript keeps everything the person saw and gains a checkpoint where
    it happened. Mirrors vibe/app_server/_handler.py `_compact`: refused
    while a turn runs, COMPACTION_FAILED when no summary came."""
    projector = _idle_session(ctx, params.session_id, "compacting")
    try:
        # The summary is a blocking model call: off the event loop.
        summary = await asyncio.to_thread(
            ctx.agent.compact, params.extra_instructions)
    except CompactionFailed as exc:
        raise RpcError(ProtocolErrorCode.COMPACTION_FAILED, str(exc))
    # Every message before the summary is gone from the model's memory, so
    # no anchor points at a message that is still there.
    ctx.anchors.clear()
    ctx.stats.last_prompt = None
    stamp = now_ms()
    checkpoint = PublicCheckpointEntry(
        id=str(uuid4()), session_id=projector.session.id,
        created_at=stamp, updated_at=stamp,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        kind="compaction", message="Context compacted",
        details={"summaryLength": len(summary)})
    projector.adopt(projector.session.model_copy(update={"updated_at": stamp}),
                    [*projector.history, checkpoint], projector.turns)
    saved.save(ctx)
    return SessionCompactResponse(
        summary=summary, state=projector.state(),
        session_log=views.session_log(projector.session.id, ctx.store))


@routes.method("session/fork", SessionForkParams)
async def session_fork(ctx, params: SessionForkParams):
    """/branch: a copy of the whole conversation under a new id, model
    memory included. `attach` continues in the copy; otherwise the copy is
    saved and the open conversation carries on."""
    projector = _idle_session(ctx, params.source_session_id, "branching")
    if params.entry_id is not None or params.after_turn_id is not None:
        raise RpcError(ProtocolErrorCode.INVALID_PARAMS,
                       "OVAT branches a whole conversation; /rewind goes "
                       "back to one message.")
    saved.save(ctx)
    copy = {**saved.record(ctx), "id": str(uuid4()), "created_at": now_ms()}
    if params.attach:
        session = projector.session.model_copy(update={
            "id": copy["id"], "created_at": copy["created_at"]})
        projector.adopt(session, _owned(projector.history, session.id),
                        _owned(projector.turns, session.id))
        saved.save(ctx)
        state = projector.state(history_limit=params.history_limit)
    else:
        if ctx.store is None:
            raise RpcError(ProtocolErrorCode.CONFLICT,
                           "Nothing is saved here, so there is nowhere to "
                           "put a branch.")
        ctx.store.save(copy)
        state = _saved_state(ctx, copy, params.history_limit, 200)
    return SessionForkResponse(source_session_id=params.source_session_id,
                               state=state,
                               last_event_id=projector.event_id)


NATIVE_ONLY = ("This needs the native engine: it is the one whose memory "
               "OVAT can cut. `/engine native` switches.")


def _idle_session(ctx, session_id: str, doing: str):
    projector = ctx.require_session(session_id)
    if ctx.turns.busy:
        raise RpcError(ProtocolErrorCode.CONFLICT,
                       f"Wait for the current turn to end, or interrupt it, "
                       f"before {doing}.")
    if not ctx.is_ready:
        raise RpcError(ProtocolErrorCode.CONFLICT,
                       "The agent is not ready yet.")
    return projector


def _owned(items: list, session_id: str) -> list:
    """Entries or turns, relabelled as belonging to `session_id`."""
    return [i if i.session_id == session_id
            else i.model_copy(update={"session_id": session_id})
            for i in items]


@routes.method("session/log/read", SessionLogReadParams)
async def session_log_read(ctx, params: SessionLogReadParams):
    projector = ctx.require_session(params.session_id)
    return SessionLogReadResponse(log=views.session_log(projector.session.id, ctx.store))


@routes.method("session/ready/wait", SessionReadyWaitParams)
async def ready_wait(ctx, params: SessionReadyWaitParams):
    ctx.require_session(params.session_id)
    try:
        await ctx.wait_ready()
    except BuildFailedError:
        # Not an error: the copied app treats a failing wait as fatal and
        # asks the person to quit, but a workflow that does not build can be
        # fixed with /workflow. The reason was announced when the build
        # failed, and a prompt sent now fails with it too.
        return SessionReadyWaitResponse(ready=False, init_duration_ms=None)
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
        session_log=views.session_log(projector.session.id, ctx.store))


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
