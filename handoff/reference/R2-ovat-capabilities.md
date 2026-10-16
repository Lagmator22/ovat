# OVAT backend capabilities vs. a full Vibe TUI port (branch feat/a2a)

Repo: `/home/user/wt/feat-a2a`. All paths are relative to it. Line numbers are as of this read.
Scope: what the UI-free backend (`ovat/agentic/`, `ovat/agent/`, `ovat/a2a/`, providers, tools) can already feed
to `ovat agent` (`ovat/tui/app.py` + `ovat/tui/bridge.py` + copied `ovat/tui/vibe/`), and what must be built.

---

## 1. Public API

### 1.1 `AgentSession` (`ovat/agentic/session.py`)

Threading model (docstring :1-13, `_spawn` :382-386): every slow call runs on a fresh daemon thread named
`ovat-agent-session`; everything the UI sees goes through ONE callback `sink(event)` (wrapped by `_emit` :376-380,
which swallows sink exceptions). The sink is called on the WORKER thread; the TUI re-posts with `post_message`
(`app.py:468-469`). One `threading.Lock` (`_lock`) guards `busy`. Agent bus listeners also run on the agent's
thread (events.py:25-27).

Constructor `AgentSession(workflow, sink, *, load=None, build=None, close=None, cwd=None)` (:126-149). Seams:
`load` (default `load_workflow`), `build` (`build_agent`), `close` (`close_agent`).

Public attributes: `workflow`, `sink`, `cwd`, `config`, `agent`, `info` (dict from `describe_agent`), `broker`
(`ApprovalBroker`, :140-141), `busy`, `last_answer`, `last_trace`, `totals` = `{"turns", "prompt_tokens",
"completion_tokens"}` (session-cumulative, :145), property `serving` (:337-339).

Methods:

| method | line | what it does | ends with |
| --- | --- | --- | --- |
| `start(engine=None)` | :153 | load + build on a worker; optional engine override via `model_copy` (:166-168); swaps the old agent out (`_drop_agent`), sets `agent.events.approver = broker`, subscribes `AgentActivity` forwarding | `Ready(info)` or `BuildFailed(text)` |
| `switch_workflow(path)` | :182 | sets `workflow`, calls `start()` | same |
| `new_conversation()` | :186-200 | forgets session grants; native: replaces `agent.session` with an empty `Session` (no reload) and emits `Notice`; framework engines: full rebuild via `start(engine)` | `Notice` or `Ready` |
| `close()` | :202 | cancel + stop A2A server + close agent | - |
| `ask(text)` | :220-230 | refuses with `Notice("warn")` if busy or no agent, else runs `_turn` on a worker | `TurnStarted` ... `TurnDone` |
| `ask_subagent(name, task)` | :232-251 | calls the `delegate` tool function directly (no main model); strips the "[Answer from subagent ...]" framing | `TurnStarted` ... `TurnDone` (no totals) |
| `cancel()` | :291-305 | sets `agent.events.stop`, `broker.cancel_all()` (deny all open), kills a running `!` process tree | returns bool |
| `run_shell(command)` | :309-333 | `!cmd` through `ovat/cli/shell.py` (`spawn`, `iter_display_lines`, `venv_env(columns=100)`) | `ShellLine`* then `ShellDone` |
| `start_server()` / `stop_server()` | :341-372 | `/serve`: spawns `python -m ovat.cli.main a2a serve <workflow>` as a subprocess; pumps its stdout as `Notice("info", "a2a: ...")` | `Notice` lines, final `Notice` with exit code |

`_turn` (:253-279): clears `stop`, emits `TurnStarted`, runs `agent.run(text)` (or the `ask_subagent` closure),
reads `agent.last_trace["totals"]`, computes `failed` (totals.failed, `agent.last_failed`, or answer starts
"Error: "), `cancelled` (= stop flag), accumulates `self.totals`, emits `TurnDone`. Sources come from the native
trace's `tool_calls[].sources` (`_sources` :281-289), so they are empty on framework engines.

`describe_agent(config, agent, workflow)` (:103-122) -> `info` dict: `workflow, name, model (folder name), device,
provider, engine, tools: [{name, kind, permission}], subagents: [{name, description, where}]`.

`find_workflow(cwd)` (:88-94): `./workflow.yml|yaml`.

Session events (frozen dataclasses, :30-82):

| event | fields |
| --- | --- |
| `Ready` | `info: dict` |
| `BuildFailed` | `text` |
| `TurnStarted` | `text` |
| `AgentActivity` | `event: ovat.agent.events.Event` (raw bus event) |
| `ApprovalNeeded` | `request: ApprovalRequest` |
| `TurnDone` | `answer, elapsed, totals: dict, sources: list, failed, cancelled` |
| `ShellLine` | `text` |
| `ShellDone` | `command, code: int|None, elapsed` |
| `Notice` | `level` (info/warn/error), `text` |

There is NO token/delta event, NO reasoning event, NO stats/usage event besides `TurnDone.totals`, NO
"conversation replaced" event, NO queue event.

### 1.2 Agent event bus (`ovat/agent/events.py`)

`EventBus` (:72-116), one per built agent (factory.py:440, attached in `_own` :448-452):
- `subscribe(listener) -> unsubscribe` (:93), copy-on-write tuple of listeners.
- `emit(type_, **data)` (:104) never raises.
- `stop: threading.Event` (:83): cancel flag. Native loop checks it before each model call (loop.py:207); every
  tool wrapper checks it before running (events.py:151). Framework engines stop only at the next tool call.
- `context: dict` (:87): per-run facts (A2A `hops`, `chain`).
- `approver(name, arguments, reason) -> bool` (:91): None means `ask` is a no.

`Event(type, data, t=monotonic)` (:56-66), `to_dict()` flattens.

Event types (:37-49):

| type | data fields | emitted by |
| --- | --- | --- |
| `tool_started` | `call_id` (8 hex), `name`, `arguments` (kwargs dict) | `observed()` :154 |
| `tool_ended` | `call_id, name, status` (`ok`/`error`/`denied`), `duration_s, result_chars` | `observed()` :158, :168 |
| `remote_status` | `agent, state` ("working"), `text` | `a2a_call_impl.on_status` (a2a_call.py:91-93) |
| `subagent_status` | `agent, text` | `SubagentRunner._status` (subagents.py:161-162) |

`describe(event)` (:119-134) -> one line ("calling X", "X done in N s", "agent: text").
`observed(name, fn, bus, permission)` (:137-172) is the ONE tool wrapper (applied by `factory.observe_tools`
:373-384), so all four engines get identical tool events, the cancel check and the permission gate. Note: the
`tool_ended` event carries only `result_chars`, never the result text.

### 1.3 Permissions (`ovat/agent/permissions.py`)

- `allow | ask | deny` (:31); defaults from `default_permission(kind, url)` (:51-61): `api` asks, `a2a` asks unless
  loopback, everything else allows. Workflow override: `tools[].permission` (workflow.py:345),
  `subagents[].permission` (workflow.py:513). Applied in `factory.build_tools` :355-362.
- `gate()` (:64-88): returns None or the "Error: ..." sentence the model reads.
- `ApprovalBroker` (:102-169): `__call__` blocks the worker (timeout 600 s, :112) -> `on_request(request)`;
  `resolve(id, decision)` (:141), `cancel_all()` (:149), `pending()` (:156), `forget_session_grants()` (:160),
  `auto_approve` (`/yolo`, :114). Decisions: `ONCE`, `SESSION` (per tool name, in memory), `REFUSE` (:35-37).
- `ApprovalRequest(id, tool, arguments, reason, decision)` (:91-99).
- There is NO persistent ("always, saved to disk") grant and NO rule list (pattern rules). The TUI maps Vibe's
  "always permanent" button to `SESSION` (app.py:384-386).

### 1.4 Subagents (`ovat/agent/subagents.py`, `ovat/tools/delegate.py`)

`subagents:` -> ONE `delegate(agent, task)` tool with an enum of names (delegate.py:40-49, factory.py:363-369,
kind `subagent`, permission allow; each subagent is gated inside `run`). `SubagentRunner.run` (:58-81): gate with
reason "runs on this machine / sends the task to URL", `subagent_status` "started (kind)", run, "finished in N s".
- workflow (:85-111): builds the child per task (fresh context, own `delegate` removed), shares `stop` and
  `approver`, subscribes to the child's bus and FORWARDS ONLY `describe()` text as `subagent_status` (:102, :164-167).
- api (:113-141): `OVMSLLMProvider(api_key=...)` + `AgentLoop(max_iterations=1)`, one status "asking MODEL".
- a2a (:143-157): `SDKPeer` + `a2a_call_impl`, progress as `remote_status`.
Answers are framed and cut at 16000 chars (:31-41).

### 1.5 `Session` memory (`ovat/agent/session.py`)

`Session(system_prompt)` (:16), OpenAI-format `messages` list, `add_user/add_assistant(content, tool_calls,
sources, sources_label)/add_tool_result`, `save(path)` (JSON, lock-snapshotted, :79-87), `Session.load(path)`
(:90-95). Used only by the native `AgentLoop` (`agent.session`) and by the launcher chat.

---

## 2. Capability inventory

Legend: HAVE = backend supports it and the agent TUI uses it; PARTIAL = something exists but is incomplete or only
on some engines / only in another screen; MISSING.

| capability | status | where / notes |
| --- | --- | --- |
| Answer token streaming, native engine | MISSING | `AgentLoop.run` calls `llm.chat(messages, tools=...)` (loop.py:216); `OVMSLLMProvider._ask` is a single non-stream `chat.completions.create` (llm_ovms.py:132-154). No `on_token` path. |
| Streaming, react / llamaindex / openai-agents | MISSING | LangChain `graph.invoke` (langchain_agent.py:116); LlamaIndex built with `streaming=False` on purpose (llamaindex_agent.py:239-244, AGENTS.md:62-63); Agents SDK `Runner.run` not `run_streamed` (openai_agents_agent.py:189). |
| Streaming, local genai path | PARTIAL | `GenAILLMProvider.chat(..., on_token)` streams and honors "return True to stop" (llm_genai.py:216-264); `rag_chat(..., on_token)` forwards it (rag_chat.py:69-101); the LAUNCHER chat uses it with MarkdownStream (chat_screen.py:1259-1323). The agent loop never passes `on_token`, so `ovat agent` with `provider: genai` + native does not stream. Also `build_llm` leaves genai `max_new_tokens` at its default 256 (factory.py:63-64, llm_genai.py:175). |
| Reasoning / thinking text | PARTIAL | `split_thinking` (chat_screen.py:97-119) + `ui.strip_thinking`; `/thinking` in the launcher chat only. Config `model.enable_thinking` (workflow.py:76) and `reasoning_parser` (workflow.py:69, passed to `ovms --reasoning_parser`, model_server.py:217). OVMSLLMProvider ignores any `reasoning_content` field (not verified that OVMS sends it). Agent TUI: raw answer goes to `bridge.answer` with no strip (app.py:400); CLI strips at main.py:397. Vibe handler already renders `PublicReasoningEntry` deltas (event_handler.py:247-251, :373-394). |
| Per-turn token usage | HAVE (end of turn) | Native per model call in `last_trace.turns[]` (loop.py:217-224) and totals (:171-204); frameworks via `usage.framework_trace` (usage.py). Delivered in `TurnDone.totals`; session sum in `AgentSession.totals`; bar shows the sum (app.py:285-289); `/trace` shows per-turn (app.py:533-553). Absent stays None (AGENTS.md:507). Subagent and A2A child tokens are NOT counted. |
| Context window size / fill | MISSING | No config field for context length; only NPU `model.ovms_max_prompt_len` (workflow.py:226). Last turn's `prompt_tokens` is the best proxy for "context used". Vibe's `ContextProgress` widget and `AgentStatsSnapshot.context_tokens` (models.py:442-470) wait for data. |
| Cancel | HAVE | `AgentSession.cancel` (:291); Esc (`action_interrupt` app.py:631) and Ctrl-C. Native stops before next model call; frameworks at next tool call (AGENTS.md:580-582). The in-flight OVMS request is not aborted. |
| Conversation save / load | PARTIAL | `/save`, `/load` (app.py:575-600) on native only (`agent.session`); `/load` does not redraw the transcript. Files in `.ovat/sessions/*.json`, the SAME folder and format as the launcher chat. |
| List / pick / resume sessions | PARTIAL | `list_sessions(cwd)` (name, path, modified, turns, first question) and `SessionPicker` live in `ovat/cli/chat_screen.py:150-258` (textual module); agent TUI's `/load` without a name only lists the last 8 stems (app.py:590-593). No autosave in the agent TUI (the launcher autosaves each turn, chat_screen.py:1313-1316). No "resume last" at startup. |
| New conversation | HAVE | `new_conversation` (:186-200); framework engines rebuild the whole agent (model/MCP reload) because their history lists are private (`_messages`, `_history`, `_input_items`). |
| Rewind / fork to an earlier turn | MISSING | Native `Session.messages` is a plain list, so truncation is trivial; framework histories are private per adapter. No API, no event. |
| Switch model at runtime | MISSING | No API. `OVMSLLMProvider.max_tokens` is read fresh per call (llm_ovms.py:47-51) so the answer cap can be retuned live; the launcher has `/tokens`. A model switch = `config.model.name` override + rebuild (OVMS must serve that model; not verified for multi-model OVMS configs). |
| Switch engine | HAVE | `/engine` -> `session.start(engine)` (app.py:525-531, session.py:166-168). History is lost (each engine has its own memory). `genai` + non-native is rejected at build (factory.py:423-429). |
| Switch workflow | HAVE | `/workflow path` -> `switch_workflow` (app.py:565-570). |
| List tools with permissions | HAVE | `info["tools"]` -> `/tools` table (app.py:484-492). MCP tools are listed per tool, not grouped by server. |
| Approvals once / session / refuse | HAVE | `ApprovalBroker` + `ApprovalApp` (app.py:339-389); queue of pending requests (app.py:335-337). `/yolo` = `auto_approve`. |
| Permission rules (persistent, patterned) | MISSING | Only per-tool `permission:` in YAML and in-memory session grants. |
| List subagents | HAVE | `info["subagents"]` -> `/agents` (app.py:494-508), includes A2A peers used as tools and the local server state. |
| Delegate to a subagent | HAVE | model-driven `delegate`, or `/ask name task` (`ask_subagent`). |
| Subagent live progress | PARTIAL | Only `subagent_status` text lines ("calling X", "X done in N s"), shown as the latest line under the running card. Structured child events, child answer stream, child tokens are dropped (see section 4). |
| A2A serve | HAVE (as subprocess) | `/serve` (:341-372). Status: only stdout lines as Notices and "a2a serving" in the bar. Server has `GET /_ovat/status` (sdk/server.py:44, :231) returning `A2AService.status()` (service.py:248-251), behind the bearer token in `.ovat/a2a/token` (tokens.py). Not polled by the TUI. |
| A2A call | PARTIAL | As a tool (`type: a2a`, factory.py:222-261) or an a2a subagent; CLI `ovat a2a call` (a2a_cmd.py:144). No direct TUI command. |
| A2A card | PARTIAL | `card_dict` (card.py:28) + `ovat a2a card` (a2a_cmd.py:128); `SDKPeer.card()` -> `RemoteCard` for peers (client.py:103). TUI shows neither. |
| Remote status events | PARTIAL | `remote_status` with text only. The server sends structured `metadata.ovat` (public_event: type/name/status/duration, service.py:62-68, server.py:85-86) and final `trace` + `sources` (server.py:96-98); `SDKPeer._send` keeps only the LAST metadata as `RemoteReply.metadata` (client.py:192-194) and passes only text to `on_status` (:195-196); `a2a_call_impl` ignores `reply.metadata`. |
| A2A audit log | PARTIAL | `AuditLog.tail(n)` (audit.py:42) exists, unused by the TUI. |
| MCP servers list / status | MISSING (data partly there) | `agent.mcp_servers` holds `MCPStdioServer` AND `SDKPeer` objects (factory.py:437, :241-242). `MCPStdioServer` has `command`, `tools`, private `_session`/`_error`/`_thread` (mcp_client.py:41-49), no public status/health or reconnect. A server that fails to start fails the whole build. |
| Shell commands (`!`) | HAVE | `run_shell` + `BashOutputMessage` (app.py:445-453, :408-419); Esc kills the process tree. |
| File @-mentions | PARTIAL (UI only) | Vibe `PathCompleter` is always on in `ChatInputContainer` (container.py:95-102), so `@path` completes; the backend passes the text verbatim (no file inlining, no size cap). Tools resolve misspelt paths via `tools/fuzzy.py`. |
| Image input | PARTIAL | `describe_image(image_path, prompt)` tool with a local VLM (factory.py:201-208, tools/describe_image.py) reachable only if the model calls it with a path. No image attachment on the user message (Vibe `ImageContentBlock`, `paste_path.py`, `/paste-image` exist in the copy). Not exposed over A2A by default (exposure.py PATH_TOOLS). |
| Audio input / voice | PARTIAL | `transcribe_impl(path, language, pipeline, model, device)` (transcribe.py:113) with a cached `WhisperPipeline` per (model, device) (`_load_pipeline` :85-110); needs a 16 kHz mono 16-bit WAV (`_read_wav` :47-67). No microphone capture anywhere and no audio dependency in the project. Vibe's `VoiceManagerPort` protocol is copied (voice_manager_port.py) with no implementation. |
| Telemetry | HAVE | `/telemetry` pushes `TelemetryView` (app.py:622-624, tui/telemetry.py); bottom bar CPU/NPU/RAM via `HardwareMonitor` every 2 s (agentic/hardware.py:14-47, app.py:264-277). The view's default sources omit `AgentTraceSource` (tui/telemetry.py:72-77), while the CLI includes it (main.py:667). |
| Doctor | HAVE | `/doctor [copy]` mounts `DoctorReport` (app.py:609-620, tui/widgets/doctor.py). |
| Compaction / summarizing long history | MISSING | Native `Session` grows without bound; only `rag_chat` caps history at 8 messages (rag_chat.py:95). No summarize step on any engine. Vibe renders compaction checkpoints (event_handler.py:220-234, :543-552). |
| Message queueing while busy | MISSING | `ask` refuses when busy (session.py:222-225); TUI puts the text back in the input (app.py:437-441). Vibe container has queue hooks (container.py:75-77) unused. |
| Theme | PARTIAL | One theme `OVAT_THEME` registered (app.py:179-180); colors from `ui.PALETTE`. Textual's default command palette (Ctrl+P) may offer theme switching (not verified in this app); no `/theme`, nothing persisted. |
| Copy last answer | HAVE | `/copy` (app.py:602-607); Ctrl-C copies a selection. |
| Trace | HAVE | `/trace` (app.py:533-553). |

---

## 3. Gaps: smallest place to add each, and which landmines apply

General rule for all of these (AGENTS.md:193-197, :263-272): new backend state goes in `ovat/agentic/` (textual-free,
a test scans it) or `ovat/agent/`; new session events are small frozen dataclasses next to the existing ones in
`agentic/session.py` and are drawn in `app.py` via `_on_<ClassName>` (the dispatch is by class name, app.py:304-309);
new agent-level events are a new type constant in `events.py` (listeners ignore unknown types, :37-40), and the
mapping to Vibe history events stays in `bridge.py`. Every new command goes in `agentic/commands.py` COMMANDS
(single source; the registry is built from it, app.py:66-76) plus a `_cmd_<name>` method.

### 3.1 Token streaming (biggest gap)

- Smallest native path: add an optional `on_delta` to the provider contract the way `GenAILLMProvider.chat(...,
  on_token)` already does, and in `OVMSLLMProvider._ask` use `stream=True` when a callback is given, rebuilding
  `content`, `tool_calls` (accumulate `delta.tool_calls` by index) and `finish_reason`. `AgentLoop` emits a new bus
  event (e.g. `answer_delta {text}` and `reasoning_delta {text}`) per chunk; `bridge.py` turns them into
  `HistoryEntryAdded(PublicMessageEntry IN_PROGRESS)` then `HistoryEntryUpdated` with an `append` patch on
  `/content/0/text`, which the copied handler already consumes (event_handler.py:241-246). `TurnDone` then must
  not add a second final answer (app.py:398-401).
- Landmines:
  - Usage on streamed replies: OVMS documents `stream_options.include_usage` for continuous-batching servables only,
    and as not working on stateful (NPU) ones (AGENTS.md:62-63, llamaindex_agent.py:239-243). Streaming would lose
    `prompt_tokens/completion_tokens` on NPU; keep "absent is not zero" (AGENTS.md:507). Consider streaming only
    when not NPU, or keep a non-stream fallback.
  - Tool-call decoding: the loop dispatches on the PAYLOAD (AGENTS.md:34-36, loop.py:227-241). With streaming the
    tool calls arrive as deltas; `finish_reason` may be "stop", "tool_calls" or "unknown" on NPU (AGENTS.md:459-465).
    The `undecoded_tool_call` / `truncated` / `empty_answer` checks (loop.py:268-334) must run on the assembled text.
    Text streamed before the loop finds it was raw `<tool_call>` markup would already be on screen: buffer until it
    is clear the reply is prose, or retract it.
  - Markdown: never mix `stream.write()` and `update()` on one widget; tags arrive split across tokens (`<th` +
    `ink>`) (AGENTS.md:551-554). The launcher's on_token (chat_screen.py:1266-1295) is the reference: strip thinking
    on the accumulated text, append only if the visible answer still extends, otherwise retire the stream. Prefer
    splitting reasoning vs answer in the BACKEND so the bridge emits two clean entry types.
  - Framework engines: react could use LangGraph `stream(..., stream_mode="messages")`; llamaindex `stream_events`
    is already iterated (llamaindex_agent.py:189-193) but with `streaming=False`; openai-agents has
    `Runner.run_streamed`. Both async engines own their event loop and refuse a running one (AGENTS.md:70-71); they
    run on AgentSession's worker thread, so callbacks must be thread-safe (the sink already is). Each has to be
    verified against its own source first (AGENTS.md:296-318). Lower priority than native.
  - A cancelled stream: closing the HTTP stream may actually stop OVMS generation, unlike today's "the request
    finishes anyway" (loop.py:85-88). Not verified; measure before claiming.

### 3.2 Reasoning text

- Smallest: in the native loop, split the reply with the existing `split_thinking` logic. Problem: it lives in
  `ovat/cli/chat_screen.py` (a textual module). Move it next to `strip_thinking` (AGENTS.md says `ovat/text.py` is
  the shared reading-model-output module, AGENTS.md:147-150) and import it from both. Also read
  `message.reasoning_content` in `OVMSLLMProvider._ask` when a `reasoning_parser` is set (OVMS field name not
  verified). Emit `reasoning_delta` / put `reasoning` in `TurnDone`; bridge -> `PublicReasoningEntry`.
- Landmines: Qwen3/3.5 thinking is decided by the TEMPLATE; with `enable_thinking: true` the reply has no opening
  `<think>` (AGENTS.md:474-481). Thinking stays off by default; `enable_thinking: false` broke native and
  openai-agents on OVMS 2026.2.1 (AGENTS.md:429-437). The agent TUI currently shows raw `<think>` text in answers
  (no strip at app.py:400): a small, real bug.

### 3.3 Usage stats and context window

- Per-turn usage exists; add a `Stats` session event after each turn (or fold into `TurnDone`) with
  last-turn prompt/completion, session totals, tokens/s (completion_tokens / latency), and last prompt_tokens as
  "context used". Map to Vibe's `AgentStatsSnapshot` / `StatsUpdated` in bridge.
- Context size: add an optional `model.context_window` field (StrictModel: schema + PURPOSE entry in
  `scripts/gen_workflow_reference.py` + regenerate docs + test, AGENTS.md:41-44), or derive it for a local folder
  from `config.json` (`max_position_embeddings`) via `core/model_scout.py`. For NPU use `ovms_max_prompt_len`.
  Do not show 0 when unknown.

### 3.4 Sessions: autosave, list, resume, rewind, fork

- Move `list_sessions` / `_session_path` out of `ovat/cli/chat_screen.py` into a textual-free module (e.g.
  `ovat/agentic/sessions.py`, or `ovat/agent/session.py`) so both screens share ONE implementation (rule 5,
  AGENTS.md:273-279). Add `AgentSession.save(name)`, `load(name)`, `list_saved()`, autosave after each `TurnDone`
  (pattern: chat_screen.py:1313-1316).
- Redraw on load: emit a `ConversationReplaced(messages)` event; bridge converts stored user/assistant/tool messages
  into history entries (tool messages carry `tool_call_id`; assistant messages carry `tool_calls`, `sources`).
- Rewind/fork (native): `Session.messages[:k]` truncation at a user-message boundary, then save as a new name for
  fork. Keep the history alternating (loop.py:208-212 explains why).
- Framework engines: their histories are private and engine-specific (`_messages` LangChain objects,
  `_history` LlamaIndex ChatMessages, `_input_items` SDK items). Cheapest consistent fix: give each adapter a
  `reset()` (so `new_conversation` stops rebuilding the model) and leave save/load/rewind native-only, said
  plainly in the UI. Converting to/from the OpenAI dict format per engine is larger.
- Note: the launcher's sessions hold `rag_chat` turns (no tool messages) and the agent's hold full tool-call
  history; both are in `.ovat/sessions/`. A picker should show which kind, or use separate folders.

### 3.5 Model switching

- `AgentSession.start(engine, model=None)` using the same `model_copy` override as engine (session.py:166-168).
  For OVMS that only works if the server serves that model; there is no API to list OVMS-served models in the
  backend (`core/model_manager.py` wraps `ovms --list_models` for the repository, not the running server;
  `/v3/models` support not verified). For genai, `core/model_scout.find_models` / `CHAT_KINDS` list local folders.
- Rebuild cost: it reloads the embedder and restarts MCP servers; history is lost unless 3.4's carry-over is done.

### 3.6 Message queueing

- Smallest: a `deque` in `AgentSession`; `ask` appends when busy and emits `QueueChanged(items)`; `_turn`'s end pops
  and starts the next one. Cancel should offer to clear the queue. The Vibe container already has
  `queue_items_getter` etc. (container.py:75-77). The approval broker and the `busy` lock already make one turn at
  a time safe.

### 3.7 Compaction

- Native only, smallest: `AgentSession.compact(instructions)` runs one extra `llm.chat` with a summarize prompt over
  `Session.messages`, then replaces them with `[system, summary-as-user/assistant pair]`. Trigger manually
  (`/compact`) first; auto-trigger needs 3.3's context size. Emit start/end so the bridge can produce a
  `PublicCheckpointEntry(kind="compaction")`.
- Landmines: the summary call itself is bounded by `model.max_tokens` (4096) and, on NPU, by the static prompt cap
  (AGENTS.md:461-466). A tool-call message pair cut in half makes the next OVMS request malformed (loop.py:52-58).

### 3.8 MCP list / status

- Add a small public `status()` to `MCPStdioServer` (connected = `_session is not None`, `error`, `command`,
  tool names) and put an `mcp` list into `describe_agent` by grouping `tools` by `_entry` before it is popped
  (factory.py:355-362) or by walking `agent.mcp_servers` and filtering `isinstance(..., MCPStdioServer)`. A
  `/mcp` command reads it. Restart/reconnect is a separate, larger change (one manager coroutine owns connect and
  unwind, AGENTS.md:126-130).

### 3.9 Image and voice input

- Image: smallest is "attachment = path": when the prompt carries `@image.png`, prepend a hint so the model calls
  `describe_image`, or pre-run `describe_image_impl` and inline the description (no model change). A real
  multimodal message needs a VLM servable and content-part messages on OVMS: not verified, and the native loop's
  messages are text-only.
- Voice: implement Vibe's `VoiceManagerPort` in a textual-free `ovat/agentic/voice.py`: capture mic audio (needs a
  new optional dependency, e.g. `sounddevice`, behind an extra, rule 4), write 16 kHz mono int16 WAV (or pass the
  float array straight to the cached pipeline, skipping `_read_wav`), call `transcribe._load_pipeline`. Whisper
  model path/device come from the workflow's `transcribe` tool entry (`model`, `device`) or `OVAT_WHISPER_MODEL`.
  Run on a worker thread (pipeline load takes seconds).

### 3.10 A2A status and cards in the TUI

- Poll `GET http://127.0.0.1:<port>/_ovat/status` with the token from `.ovat/a2a/token` (or have `start_server`
  read the port from the workflow's `a2a.server`) to show active/queued/done counts; tail the audit log with
  `AuditLog.tail`. Show peers' cards: `SDKPeer.card()` is already fetched at build (factory.py:243-248) but the
  `RemoteCard` is discarded; keep it on the tool spec (e.g. `spec["card"]`) and put it in `describe_agent`.
- Landmine: only `ovat/a2a/sdk/` may import `a2a` (AGENTS.md:168-173, :271-272); the TUI must go through
  `SDKPeer`/`protocol.py` shapes.

### 3.11 Permission rules

- Persisted grants: store `{tool: allow}` in `.ovat/permissions.json` and preload `broker._session_allowed`; or add
  rules (tool + argument pattern) evaluated in `ApprovalBroker.__call__` before asking. Keep `gate()`'s
  "no approver means no" rule untouched (AGENTS.md:182-186): rules must only ever apply where a broker exists.

### 3.12 Small fixes found while reading

- Agent TUI shows raw `<think>` reasoning in answers (app.py:400, no `strip_thinking`).
- `/telemetry` from the agent TUI omits `AgentTraceSource` (tui/telemetry.py:72-77) although the session has an
  agent (`TelemetryView(sources=...)` can take one).
- `ovat agent` with `provider: genai` caps answers at 256 tokens (factory.py:63-64 never passes
  `max_new_tokens`); the launcher deliberately runs uncapped (AGENTS.md:482-489).
- `bridge.progress` attaches progress to `_last_running` (bridge.py:139), i.e. whichever tool started last. Fine on
  the native loop (sequential tools), wrong if a framework engine runs tools in parallel; key progress by agent
  name instead.

---

## 4. Subagents and A2A in the TUI today

Mapping in `ovat/tui/bridge.py`:
- `tool_started` (bridge.py:89-102): a `PublicEffectEntry` with `RunningEffectState`. If the tool kind is `subagent`
  or `a2a` (kinds come from `info["tools"]`, app.py:328-329), the detail is a `SubagentEffectDetail` with
  `agent_name = arguments["agent"]` (delegate) or the tool name (a2a), `task = arguments["task"]` or
  `["message"]`, verbs "Delegating to / Delegated to" (bridge.py:66-75). `search_docs` -> FileSearch,
  `transcribe`/`describe_image` -> FileRead, else Generic with the arguments (:76-87).
- `tool_ended` (:104-134): Completed (ok), Skipped "not allowed" (denied), Failed (error). The result TEXT is
  never shown: the bus event has only `result_chars`.
- `subagent_status` / `remote_status` (:136-148): a `HistoryEntryUpdated` with an `append` patch on
  `/state/outputText` for the last running call. The handler calls `ToolCallMessage.set_stream_message(output)`
  (event_handler.py:252-256), which REPLACES the line under the card with "-> agent: text" (tools.py:429-433) and
  clears it on settle (:435-440). So only the latest progress line is visible, and nothing remains afterwards.
  The patch is not applied to the entry, so the completed card has no record of the progress either.
- `/agents` (app.py:494-508): static list of subagents, A2A tools and whether `/serve` runs. `/ask` routes to
  `ask_subagent`. `/serve` output is shown as info notices "a2a: ...".
- Approvals for a subagent are titled "subagent NAME" with the task as the argument and the reason
  ("sends the task to URL") (subagents.py:69-74); shown through `approval_safe` (app.py:94-115).

Available but not shown:
- Workflow subagent: the child's full bus (tool_started with arguments and call_id, tool_ended with status and
  duration, and its own nested events), the child's `last_trace` (turns, tokens, flags such as
  `undecoded_tool_call`), its full answer before the 16000-char cut, and which workflow/engine/model it ran.
  `SubagentRunner._forward` reduces all of it to `describe()` text (subagents.py:164-167). Smallest change: also
  re-emit the child event as a new type (e.g. `subagent_event {agent, event: child.to_dict()}`) and a
  `subagent_done {agent, trace totals}`; bridge nests them as child effect entries (Vibe has
  `ChildSessionUpdated` and `subagent_list.py` for this) and adds the child tokens to the session totals.
- API subagent: the model name, the base URL, and its usage (`loop.last_trace`), all available in `_run_api`
  (subagents.py:113-141); only "asking MODEL" is emitted.
- A2A remote: per-update structured events from the remote OVAT (`metadata.ovat` = public_event dict with type,
  name, status, duration_s, result_chars; arguments removed by `public_event`, service.py:62-68), the queue
  position (`{"type": "queued", "ahead": n}`, service.py:279-281), and the final `trace` totals and `sources`
  (server.py:96-98). `SDKPeer._send` keeps only the last metadata and passes text to `on_status`
  (client.py:186-196); `a2a_call_impl` discards `reply.metadata` (a2a_call.py:111-124). Also the remote's card
  (name, skills, version) and `task_id`/`context_id` (conversation continuity, `SDKPeer.reset()` exists at
  client.py:119 but nothing calls it on `/new`).
- Local A2A server: live counts (`/_ovat/status`), the audit log of what left the machine, the card it serves,
  exposed/hidden tools and warnings (`Prepared.exposure`, printed only to the subprocess stdout).
