# 01. Architecture of the ported TUI (read before writing any code)

## 1. The one idea

Mistral Vibe's TUI never touches an agent. Its app class (`VibeApp`,
`ovat/tui/vibe/cli/textual_ui/app.py`) talks only to an `AppServerSession`
(`ovat/tui/vibe/app_server/session.py`) and its `resources`, which are
JSON-RPC CLIENTS. They send requests over a transport to an "app server"
and rebuild a mirror of the session from the notifications that come back.

Vibe ships its own server, written for Vibe's agent. We did NOT copy it.
OVAT implements the SERVER side itself, in `ovat/tui/server/`, on top of
`ovat.agentic.session.AgentSession`. So:

```
 OvatAgentApp (subclass of VibeApp, ovat/tui/agent_app.py)
   |  widgets, screens, slash commands: unchanged Vibe copies
   v
 AppServerSession + resources (copied client, ovat/tui/vibe/app_server/)
   |  JSON-RPC 2.0 over memory_transport_pair() (in-process, no socket)
   v
 RpcServer (ovat/tui/server/rpc.py)  ->  methods/*.py handlers
   |                                         |
   v                                         v
 ServerContext (context.py) ---------> Projector (projector.py): the TRUE
   owns AgentSession, TurnManager,       session state, numbers every event,
   Approvals, ShellRuns                  sends notifications to the client
   ^
   |  sink callback, from worker threads, hopped onto the loop with
   |  loop.call_soon_threadsafe (context._sink)
 AgentSession (ovat/agentic/session.py): builds the agent on a worker
   thread, runs one turn at a time, reports ToolStarted/ToolEnded/
   TurnDone/ShellLine/ShellDone/... events
```

The contract is typed and copied, so it IS the spec:
- every method's params and response: `ovat/tui/vibe/app_server/protocol.py`
- notifications (events): `ovat/tui/vibe/app_server/events.py`
- history entries and effects: `models.py`, `_effect_models.py`
- what the client actually calls: `ovat/tui/vibe/app_server/resources/*.py`
  (the `_runtime_resources.py`, `_session_resources.py` etc. files)

Tests drive OVAT's server through Vibe's REAL client, so they are contract
tests, not mocks we wrote: `tests/test_ovat_app_server.py` (70 tests at 2a).

## 2. Files and what each owns

### OVAT's own code (edit freely, keep style)

| File | Owns |
|---|---|
| `ovat/tui/server/__init__.py` | `start_ovat_app_server(workflow, *, session_factory=AgentSession, cwd=None, continue_session=False, resume_session_id=None, client_info=None, capabilities=None) -> (AppServerSession, OvatServerHandle)`; `serve_on_memory`; `OvatServerHandle.agent_session`, `.close()` |
| `ovat/tui/server/rpc.py` | framing, dispatch by method name, params validation with the protocol model, errors, server->client requests (callbacks). `Routes` = name -> (params model, handler). `not_supported(method)` gives `not_implemented` for names Vibe's own server knows (`SERVER_METHODS`), `method_not_found` otherwise; message is "not supported in OVAT: <method>" |
| `ovat/tui/server/context.py` | `ServerContext`: one conversation. The AgentSession, the Projector, the TurnManager, Approvals, ShellRuns. `_sink` is the ONLY place a worker-thread report crosses to the loop. `ctx.require_session(id)` validates the session id. `ctx.info` has model/provider/device |
| `ovat/tui/server/projector.py` | the true public session: history entries, turns, queue, callbacks. Every change is a numbered event (`eventId`); the client accepts only the next number and re-reads on a gap. Entries change by small JSON patches (append to answer text, replace a tool state), never rewritten whole |
| `ovat/tui/server/turns.py` | `TurnManager`: the running turn and the queue. Event order matters and mirrors Vibe's server: response, queueUpdated (added), queueUpdated (promoted), turn/started, user message, agent entries, stats, turn/completed. A prompt sent while the model is still loading is accepted and waits |
| `ovat/tui/server/approvals.py` | `ApprovalBroker` (worker thread blocks) <-> Vibe `callback/call` / `callback/result`. approve -> ONCE, approve for session -> SESSION, approve permanently -> SESSION (no rules on disk yet), deny -> REFUSE, cancel -> REFUSE + interrupt |
| `ovat/tui/server/shell_runs.py` | `!command` via `AgentSession.run_shell`, output batched into `/state/outputText` appends |
| `ovat/tui/server/views.py` | `runtime/read`, `config/read`, stats pictures. Vibe config fields OVAT does not have are filled with "off" values HERE and only here. `/yolo` shows as the second agent profile |
| `ovat/tui/server/safe.py` | `approval_safe`: clean model-written text before a person decides on it |
| `ovat/tui/server/bridge.py` | OVAT tool/subagent/remote events -> `PublicEffectEntry` / `PublicMessageEntry` (`ovat/tui/bridge.py` is now only a re-export) |
| `ovat/tui/server/methods/{lifecycle,session,turn,runtime,config,host}.py` | one handler per method, each module exposes `routes = Routes()`; `methods/__init__.py:all_routes()` merges them |
| `ovat/tui/agent_app.py` | `OvatAgentApp(VibeApp)`: OVAT banner, OVAT slash commands, registry filtered to what the server supports (Phase 2b) |
| `ovat/tui/agent_commands.py` | the OVAT slash commands and their words, shared between the two apps (Phase 2b) |
| `ovat/tui/app.py` | the CLASSIC OVAT agent TUI (pre-Vibe). Still present as fallback |
| `ovat/tui/widgets/doctor.py`, `ovat/tui/telemetry.py` | `/doctor` block and `/telemetry` screen (work in both apps) |
| `ovat/text.py` | `split_thinking` (one copy; strips `<think>` blocks) |
| `ovat/agentic/session.py` | `AgentSession`: the agent behind every UI. `TurnDone.reasoning` field added |

### Copied Vibe code (do NOT restyle; edit only when unavoidable)

`ovat/tui/vibe/**` (229 files). Allowed edits so far: the import rename,
3.10 shims through `ovat/tui/vibe/_compat.py`
(StrEnum, UTC, Self, assert_never, override, tomllib fallback,
naturaldelta/naturalsize, toml_dump, task_cancelling, fromisoformat), branding
strings, and the few hooks Phase 2b needed in `cli/textual_ui/app.py` and
`observability/sentry.py` (Sentry must never send anything). Every edit
beyond the rename must be small and listed in the commit body.

Styles: five `.tcss` files ship via `package-data` in `pyproject.toml`.
Textual gotchas learned the hard way:
- the APP's CSS outranks any widget's `DEFAULT_CSS`;
- a Screen's CSS is scoped to DESCENDANTS unless `SCOPED_CSS = False`;
- Vibe's `app.tcss` makes `Screen` transparent and `Horizontal` height auto,
  so any OVAT screen must set its own background and heights with ids
  (see `ovat/tui/telemetry.py` for the pattern).

`pyproject.toml` `tui` extra: textual, pyfiglet, anyio, certifi, httpx,
jsonpatch, keyring, packaging, pyperclip, truststore, tomli (py<3.11).

## 3. Supported methods at end of Phase 2a (40)

initialize / initialized; session/start, read, stop, list, log/read,
ready/read, ready/wait, settings/update, agent/update, history/list,
history/clear, shellCommand; turn/start, turn/interrupt,
session/turn/enqueue, queue/read, queue/replace, queue/remove, queue/resume;
callback/result; runtime/read, stats/read, tools/list, agents/list,
diagnostics/list; config/read, config/effective/read, config/schema;
account/read, identity/read, providerAuth/read, feedback/shouldShow,
feedback/record, telemetry/record (accepted and dropped, nothing leaves the
machine), workspace/trust/status, workspace/trust/untrustedConfig,
workspace/prompt/prepare.

Partial: `session/resume` works only for the SAME session id (no saved
sessions yet); `session/continue` returns an error until Phase 3.

Full list of all 100 client calls with their params/response models and the
resource file that calls each: `reference/R5-rpc-method-table.txt` (`OK` =
answered by OVAT, `--` = not yet). Regenerate it with:

```bash
SUP=$(.venv/bin/python -c "from ovat.tui.server.methods import all_routes; print(','.join(all_routes().names()))")
# then grep the resources for each method string; any line-based script works
```

Never implement (product decisions, not gaps): `vibeCode/*`,
`connector_catalog/*`, `connectors/*`, `plugin*`, `skills/*` (unless OVAT
grows skills), `narration/*`, `review/*`, anything Mistral-cloud or account
related. `telemetry/record` stays a no-op (no phoning home).

## 4. Recipe: add (or finish) one RPC method

1. Find the client call: `grep -rn '"session/rename"' ovat/tui/vibe/app_server/`.
   Note the params model, the response model and which event(s) the client
   expects afterwards (read the resource method and `session.py`'s event
   handling).
2. Read the models in `protocol.py` / `events.py` / `models.py`. Field names
   are snake_case in Python, camelCase on the wire (`model_dump(by_alias=True)`
   is done for you by `rpc.py`).
3. Add the handler to the right `methods/*.py`:
   ```python
   @routes.method("session/rename", SessionRenameParams)
   async def session_rename(ctx, params: SessionRenameParams):
       ctx.require_session(params.session_id)
       ...                          # change state through ctx / ctx.projector
       return SessionRenameResponse(...)
   ```
   State changes that the client mirrors MUST go through the Projector so they
   get an `eventId`. Never write to the transport from a worker thread.
4. If the backend lacks the ability, add it to `AgentSession`
   (`ovat/agentic/session.py`) or lower, with its own unit test, NOT in the
   server. The server only translates.
5. Contract test in `tests/test_ovat_app_server.py`: start the server with a
   fake session (the file has `FakeSession` helpers), call through the real
   client (`session.resources...`), assert the response AND the mirrored
   session state. Prove it fails without the handler.
6. If a slash command was hidden because its method was missing, unhide it in
   the registry filter in `ovat/tui/agent_app.py` and add a pilot test in
   `tests/test_vibe_agent_tui.py`.
7. Run: `.venv/bin/python -m pytest -q tests/test_ovat_app_server.py`, then
   both full suites (3.13 and 3.10).

## 5. Threading and ordering rules (the bugs you will hit)

- AgentSession calls its sink from worker threads. Only `context._sink` may
  cross to the loop. Everything after it is single-threaded on the loop.
- Event numbering: if you emit outside the Projector, the client sees a gap,
  re-reads the whole session, and the UI flickers or duplicates entries.
- The approval worker thread is BLOCKED until `callback/result`; on
  interrupt or close you must resolve every open approval (REFUSE) or the
  thread hangs and `close()` never returns.
- `turn/interrupt` must work while the model is still loading.
- One turn at a time: `AgentSession.ask` refuses a second; the queue is how
  the TUI sends more.

## 6. How the app boots (`ovat agent [workflow]`)

1. `start_ovat_app_server(workflow, ...)` builds `AgentSession(workflow, sink=ctx._sink)`
   (model loads on a worker thread), creates the memory transport pair, starts
   `RpcServer` on the server end, starts `AppServerSession` on the client end
   (`initialize` handshake, `session/start` or resume).
2. `OvatAgentApp(..., app_server=session).run()`.
3. On exit, `OvatServerHandle.close()` stops the session, refuses open
   approvals, stops shell runs.

Fallback: the classic app (`ovat/tui/app.py`) stays available during the
owner's trial (see section 9 for the exact switch).

## 7. Test files

| File | Covers |
|---|---|
| `tests/test_ovat_app_server.py` | server contract through Vibe's client |
| `tests/test_vibe_agent_tui.py` | `OvatAgentApp` pilot tests (Phase 2b) |
| `tests/test_agent_tui.py` | classic app, doctor and telemetry (approval grace via `monkeypatch` of `_INPUT_GRACE_PERIOD_S`) |
| `tests/test_tui_vibe_imports.py` | every copied module imports on this Python |
| `tests/test_tui_vibe_compat.py` | the 3.10 shims |
| `tests/test_tui_branding.py` | no Mistral/Vibe branding visible to the user |
| `tests/test_tui_isolation.py` | CLI works without textual |
| `tests/test_tui.py` | launcher TUI (`_until` poll helper for ctrl_c tests) |

## 8. Screenshots for review

Textual can save SVG screenshots in pilot tests (`app.save_screenshot()`).
Convert to PNG to look at them (any SVG rasterizer; we used a small Node
script with Playwright's Chromium). Always LOOK at a new screen before calling
it done: three of the doctor/telemetry bugs were only visible on screen.

## 9. Phase 2b (switch `ovat agent` to the Vibe app)

Done, on branch `wip/p2b` (pushed, NOT yet in feat/a2a). 4 commits on top of
43069d9: e72e6e8 (shared words in `ovat/tui/agent_text.py`), 6558a06
(`handle.switch_engine/switch_workflow`, `ServerContext.rebuild`), 60f2268
(OVAT banner via `_banner_factory`, sentry guard), 00bba76 (`ovat agent` runs
`OvatAgentApp`). Suites: 3.13 and 3.10 both 1163 passed, 15 skipped, 0 failed.
New tests: `tests/test_vibe_agent_tui.py` (65).

- Entry: `ovat agent` -> `ovat.tui.runner()` -> `agent_app.run`.
  `OVAT_AGENT_UI=classic` runs the old app.
- Copied files edited (marked "OVAT"): `vibe/cli/textual_ui/app.py`
  (`_banner_factory` class attr, 2 lines) and `vibe/observability/sentry.py`
  (return if sentry_sdk missing).
- Commands (18): help, clear(/new,/reset), copy, exit, status (OVAT
  version), theme (session only), tools, agents, ask, serve, engine, trace,
  yolo, workflow, save, load, doctor [copy], telemetry, plus `!cmd`.
  28 Vibe commands are hidden. The filter is DERIVED:
  `agent_commands.VIBE_NEEDS` lists the RPC methods each Vibe command needs;
  a command shows only when all are in `all_routes()`. So implementing a
  method in Phase 3/4 unhides its command automatically. Check VIBE_NEEDS.
  An autouse test fixture fails any test where the app calls an unrouted
  method.
- Keys reaching unsupported code are disabled: Esc Esc (rewind), Ctrl+Enter
  steer, Ctrl+Backslash debug console. Unknown `/word` warns with the nearest
  command.

### KNOWN BUG, fix FIRST (found in real-model run, not yet fixed)

The banner can stay on "loading the agent..." forever after the agent is
ready. Cause: Textual runs children's `on_mount` before the parent's. The
Banner draws "loading" on mount; if the AgentSession reports `Ready` before
`OvatAgentApp.on_mount` calls `self._handle.subscribe(...)`, the banner
never hears it (the bottom bar is fine because on_mount calls
`_update_bar()`). Seen with OVMS already running (agent builds in under a
second). Adding `self._refresh_banner()` after subscribe in `on_mount` was
NOT enough (the test below still failed); find out why (read
`_refresh_banner` near line 446 and `AgentBanner.set_state`; maybe the banner
is not attached yet or `_refresh_banner` reads something else), fix it, and
keep this test, which reproduces the bug (it fails today):

```python
def test_an_agent_ready_before_the_app_mounts_still_reaches_the_banner(
        monkeypatch):
    made = []

    def factory(workflow, sink, cwd=None):
        made.append(AppFake(workflow, sink, cwd, hold_build=True))
        return made[0]

    original = OvatAgentApp.on_mount

    async def late_mount(self):
        made[0].build_gate.set()
        deadline = time.monotonic() + 8
        while not self._handle.ready and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        original(self)

    monkeypatch.setattr(OvatAgentApp, "on_mount", late_mount)

    async def go():
        session, handle = await start_ovat_app_server(
            "w.yml", session_factory=factory, cwd=CWD)
        app = OvatAgentApp(session, handle, history_file=None, animate=False,
                           hardware=False)
        try:
            async with app.run_test(size=SIZE) as pilot:
                await until(pilot, lambda: "Qwen3.5-4B-int4-ov"
                            in banner_text(app))
                assert "loading" not in banner_text(app)
        finally:
            await asyncio.wait_for(session.close(), 10)
            await asyncio.wait_for(handle.close(), 10)
    run(go())
```
(put it in `tests/test_vibe_agent_tui.py` before
`test_nothing_that_phones_home_or_nags_is_on`).

Small cosmetic: with an absolute workflow path the banner's second line runs
off the screen; show the file name only (or shorten the path).

Then: fast-forward feat/a2a to wip/p2b (`git switch feat/a2a && git merge
--ff-only wip/p2b`), push, get PR 78 CI green.
