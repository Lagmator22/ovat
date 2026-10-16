# Phase 2: the OVAT app server

## Why this seam

Vibe's TUI (`ovat/tui/vibe/cli/textual_ui/app.py`, `VibeApp`) never touches an
agent. It talks only to `AppServerSession` (`ovat/tui/vibe/app_server/session.py`)
and its `resources`, which are JSON-RPC CLIENTS. They send requests over a
transport (`transport.py`: `memory_transport_pair()` for in-process) to a server.

OVAT implements that SERVER, over `ovat.agentic.session.AgentSession`. Then:
- `VibeApp`, `AppServerSession`, `resources`, every widget: unchanged copies.
- The contract is typed: every method's params and response are pydantic models
  in the copied `ovat/tui/vibe/app_server/protocol.py`; events in `events.py`;
  history/effect entries in `models.py` / `_effect_models.py`.
- Tests drive the OVAT server through Vibe's real client: a contract test, not
  a mock of our own making.

## Layout (all OVAT's own code)

    ovat/tui/server/__init__.py      start_ovat_app_server(...) -> AppServerSession
    ovat/tui/server/rpc.py           the loop: read a message from the server end of
                                     the memory transport, dispatch by method,
                                     validate params with the protocol model, write
                                     the response or a JSON-RPC error; send
                                     notifications (events) and server->client
                                     requests (callbacks) with ids
    ovat/tui/server/methods/         one module per group: lifecycle, session,
                                     turn, runtime, config, sessions, mcp, stubs
    ovat/tui/server/projector.py     AgentSession events -> Vibe events
                                     (TurnStarted, HistoryEntryAdded/Updated with
                                     JSON-patch appends, TurnCompleted, StatsUpdated,
                                     CallbackRequested, ChildSessionUpdated,
                                     ServerWarning/Error). Reuses ovat/tui/bridge.py's
                                     mapping of tool/subagent/remote events to
                                     PublicEffectEntry; bridge.py moves under here.
    ovat/tui/agent_app.py            class OvatAgentApp(VibeApp): OVAT banner,
                                     OVAT commands (/serve /agents /ask /engine
                                     /doctor /telemetry /trace /yolo /workflow),
                                     registry filtered to supported commands,
                                     no Mistral-cloud screens.

Boot (`ovat agent [workflow]`):
1. AgentSession(workflow, sink=server.on_session_event).
2. client_t, server_t = memory_transport_pair(); server task on server_t.
3. AppServerClient(client_t); AppServerSession.start(client, client_info=...,
   capabilities=..., continue_session=--continue, resume_session_id=--resume).
4. OvatAgentApp(history_file, app_server=session).run().
Fallback during the owner's trial: OVAT_AGENT_UI=classic runs the current
ovat/tui/app.py unchanged.

## Method groups (110 methods in the client)

Phase 2 (must work, the app boots and chats):
- handshake: `initialize` request, `initialized` notification
- `session/start`, `session/continue`, `session/resume` (resume/continue may answer
  a clear error until Phase 3 sessions exist), `session/read`, `session/stop`,
  `session/ready/wait`, `session/settings/update`, `session/agent/update`
- `runtime/read` (tools, mcp=[], skills=[], connectors=[], hooks=0, context window
  if known, issues), `stats/read`, `tools/list`, `agents/list`
- `config/read`, `config/effective/read`, `config/schema` as needed by startup
  (map OVAT's workflow + UI prefs; writes come in Phase 4)
- `turn/start`, `turn/interrupt`, `turn/steer` (error until Phase 3),
  `session/turn/enqueue`, `session/turn/queue/read` (empty queue until Phase 3)
- `callback/call` / `callback/result`: approvals from ovat.agent.permissions
  ApprovalBroker (once / session / refuse), questions later
- `session/history/list`, `session/history/clear`, `session/shellCommand` (`!cmd`)
- `providerAuth/read`, `identity/read`, `account/read` -> "none / not signed in",
  `feedback/shouldShow` -> false, `workspace/trust/status` -> trusted,
  `workspace/prompt/prepare` -> text passthrough (+ @files later)
- everything else -> JSON-RPC error "not supported in OVAT" (code chosen so the
  client shows a readable message), AND the matching slash commands removed from
  the registry so nobody reaches them.

Phase 3: session/list, delete, rename, fork, rewind(+read), log/read, compact,
context/inject, turn queue (replace/remove/resume/steer), turn/steer,
config/model/write, thinking, stats with context window, streaming deltas.
Phase 4: config/write + reload (workflow.yml, validated), mcp_catalog/* (read,
toggle for the session), diagnostics/*, loops/* (maybe), question callbacks.
Never: vibeCode/*, connector_catalog/*, connectors/*, plugin*, skills/* (unless
OVAT grows skills), narration/*, telemetry/record (no phoning home), review/*.

## Rules

- No module-level `import textual` outside ovat/tui (isolation test).
- Server code must not import Vibe's server side or vibe core (not copied).
- Every handler: params validated with the protocol model, response built with the
  protocol model, `model_dump(by_alias=True)` on the wire (check how client.py
  serializes; mirror it).
- Thread safety: AgentSession calls its sink from worker threads; hop to the
  server's event loop with loop.call_soon_threadsafe before touching transport.
- Tests: tests/test_ovat_app_server.py drives the server with Vibe's
  AppServerSession and a FakeSession; tests/test_agent_tui.py moves to the new
  app; both on Python 3.10 and 3.13.
