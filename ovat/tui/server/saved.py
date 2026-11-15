# ovat/tui/server/saved.py
"""Serialization between live app-server sessions and persisted session records.

Records managed via ``ovat.agentic.sessions.SessionStore`` store session
metadata, the UI transcript (``history``, ``turns``), underlying LLM message
memory (``memory``, for native sessions), and ``anchors`` mapping user
transcript entries to message indices for accurate rewind. Runs on the server
event loop.
"""
from __future__ import annotations

from ovat.tui.server.projector import PREVIEW_CHARS, now_ms
from ovat.tui.vibe.app_server.models import (ArchivedSessionStatus,
                                             FailedSessionStatus,
                                             IdleSessionStatus,
                                             PublicChildSession,
                                             PublicHistoryEntry, PublicSession,
                                             PublicTurn,
                                             validate_history_entry)


def record(ctx) -> dict:
    """The open conversation as a record."""
    projector = ctx.projector
    session = projector.session
    agent = ctx.agent
    return {
        "id": session.id,
        "title": session.title or "",
        "preview": session.preview,
        "created_at": session.created_at,
        "updated_at": now_ms(),
        "workflow": ctx.workflow,
        "engine": ctx.info.get("engine"),
        "history": [_dump(e) for e in projector.history],
        "turns": [_dump(t) for t in projector.turns],
        "child_sessions": {cid: _dump(cs) for cid, cs in projector.child_sessions.items()},
        "child_histories": {cid: [_dump(e) for e in entries] for cid, entries in projector.child_histories.items()},
        "memory": agent.memory() if agent is not None else None,
        "anchors": dict(ctx.anchors),
    }


def save(ctx) -> None:
    """Save the open conversation, if there is anything to save and a place
    to put it. A failure is said once, not after every turn."""
    store = ctx.store
    if store is None or ctx.projector is None:
        return
    if not ctx.projector.history:
        # Rewound to its first message: there is nothing left to keep, and
        # the old file would bring the dropped turns back on /resume.
        store.delete(ctx.projector.session.id)
        return
    try:
        store.save(record(ctx))
    except (OSError, TypeError, ValueError) as exc:
        if not ctx.save_failed:
            ctx.save_failed = True
            ctx.projector.warning(
                f"Could not save this conversation: {exc}. It still works; "
                f"it just will not be in /resume.")


def find(ctx, session_id: str) -> dict | None:
    """The saved record with this id, or the one record whose id starts
    with it (`ovat agent --resume 1a2b3c4d`, the short id /clear prints)."""
    store = ctx.store
    if store is None or not session_id:
        return None
    found = store.load(session_id)
    if found is not None:
        return found
    matches = [r for r in store.list() if r["id"].startswith(session_id)]
    return matches[0] if len(matches) == 1 else None


def public_session(rec: dict, ctx, session_id: str | None = None
                   ) -> PublicSession:
    """The session-list row (and the session itself, once adopted)."""
    active = ctx.projector.session.agent if ctx.projector else None
    return PublicSession(
        id=session_id or rec["id"], title=rec.get("title") or None,
        preview=(rec.get("preview") or "")[:PREVIEW_CHARS],
        status=IdleSessionStatus(),
        created_at=rec.get("created_at") or 0,
        updated_at=rec.get("updated_at") or 0,
        cwd=ctx.cwd, workspace_roots=[ctx.cwd] if ctx.cwd else [],
        agent=active)


def history_of(rec: dict, session_id: str | None = None
               ) -> list[PublicHistoryEntry]:
    """The saved transcript; an entry this version cannot read is left
    out rather than failing the whole conversation."""
    entries = []
    for raw in rec.get("history") or []:
        try:
            entry = validate_history_entry(raw)
        except ValueError:
            continue
        if session_id is not None:
            entry = entry.model_copy(update={"session_id": session_id})
        entries.append(entry)
    return entries


def turns_of(rec: dict, session_id: str | None = None) -> list[PublicTurn]:
    turns = []
    for raw in rec.get("turns") or []:
        try:
            turn = PublicTurn.model_validate(raw)
        except ValueError:
            continue
        if session_id is not None:
            turn = turn.model_copy(update={"session_id": session_id})
        turns.append(turn)
    return turns


def child_sessions_of(rec: dict) -> dict[str, PublicChildSession]:
    """Saved child sessions; non-terminal states are restored as idle."""
    out = {}
    for cid, raw in (rec.get("child_sessions") or {}).items():
        try:
            cs = PublicChildSession.model_validate(raw)
            if not isinstance(cs.status, (IdleSessionStatus, FailedSessionStatus, ArchivedSessionStatus)):
                cs = cs.model_copy(update={"status": IdleSessionStatus()})
            out[cid] = cs
        except Exception:
            continue
    return out


def child_histories_of(rec: dict) -> dict[str, list[PublicHistoryEntry]]:
    """Saved child histories."""
    out = {}
    for cid, entries_raw in (rec.get("child_histories") or {}).items():
        entries = []
        for raw in entries_raw:
            try:
                entries.append(validate_history_entry(raw))
            except Exception:
                continue
        out[cid] = entries
    return out


def _dump(model) -> dict:
    return model.model_dump(mode="json", by_alias=True)
