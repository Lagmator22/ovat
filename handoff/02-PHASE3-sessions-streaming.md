# 02. Phase 3: streaming, sessions, rewind, queue/steer, compact, pickers

Goal: the Vibe TUI feels live (answer text appears as it is generated) and
keeps conversations (list, resume, continue, rename, delete, fork, rewind,
compact), plus model and thinking pickers and retry.

Read first: 01-ARCHITECTURE.md sections 1, 4, 5. The deep backend analysis
for every item below is in `reference/R2-ovat-capabilities.md` section 3
(numbered 3.1 to 3.12). Items 3.12 bullets 1-3 are ALREADY FIXED on feat/a2a
(thinking stripped via `ovat/text.py:split_thinking`, AgentTraceSource on
/telemetry, genai `max_new_tokens` passed). Bullet 4 (bridge progress keyed by
last running tool) is still open; fix it in Phase 5.

General rule: backend ability goes in `ovat/agentic/` or `ovat/agent/`
(textual-free, a test scans for textual imports), with its own unit test. The
server in `ovat/tui/server/` only translates. New AgentSession events are small
frozen dataclasses next to the existing ones in `ovat/agentic/session.py`.

Do these in order. Each numbered step = one or two commits, each with tests.

---

## Step 1. Token streaming (biggest visible win)

Backend (R2 3.1, read it fully, it lists five landmines):
- `OVMSLLMProvider._ask`: when a delta callback is given, request
  `stream=True`; rebuild `content`, `tool_calls` (accumulate
  `delta.tool_calls[i]` by `index`) and `finish_reason` from chunks.
  `GenAILLMProvider.chat(..., on_token)` already streams; reuse its contract.
- Native `AgentLoop`: emit `answer_delta {text}` and `reasoning_delta {text}`
  on the agent event bus (`ovat/agent/events.py`; listeners ignore unknown
  types). Split reasoning vs answer in the BACKEND using `split_thinking` on
  the ACCUMULATED text (tags arrive split across tokens: `<th` + `ink>`).
- Do not stream text that turns out to be raw `<tool_call>` markup. Buffer
  until the reply is clearly prose, or retract.
- NPU (stateful servable): OVMS does not return usage on streamed replies.
  Keep "absent is not zero": show nothing, never 0. Simplest: no streaming
  when `device` is NPU, or a non-stream fallback.
- `undecoded_tool_call` / `truncated` / `empty_answer` checks in
  `ovat/agent/loop.py` must run on the assembled text, unchanged.
- Framework engines (react / llamaindex / openai-agents): leave non-streamed
  in this phase. Say so in the commit body.

Server:
- `AgentSession` forwards deltas as a new event (e.g. `AnswerDelta(text)`,
  `ReasoningDelta(text)`).
- `projector.py`: first delta of a turn adds a `PublicMessageEntry`
  (assistant, IN_PROGRESS) or `PublicReasoningEntry`; next deltas are
  `HistoryEntryUpdated` with a JSON patch `append` on `/content/0/text`
  (the copied event handler already consumes this; see
  `ovat/tui/vibe/cli/textual_ui/` event handler, search "append").
- On `TurnDone`: mark the entry COMPLETED; do NOT add a second answer entry.
  `turns.py` currently adds the final answer from `TurnDone`: change it to
  reuse the streamed entry when one exists.
- Throttle: batch deltas (e.g. every 50 ms) like `shell_runs.py` batches lines,
  or the client gets one notification per token.

Tests: fake provider that yields chunks; assert the client mirror shows one
assistant entry whose text grows and ends COMPLETED; a tool-call reply is not
shown as text; reasoning lands in a reasoning entry. Real-model check on the
rig (00 section 5): watch text appear live.

## Step 2. Stats and context window

R2 3.3. After each turn emit stats: last turn prompt/completion tokens,
session totals, tokens/s (completion / latency), last prompt_tokens as
"context used". Map to Vibe's `AgentStatsSnapshot` and the `stats/updated`
event (look in `events.py`). Context size: add optional
`model.context_window` to `ovat/config/workflow.py` (StrictModel; also add to
`scripts/gen_workflow_reference.py` and regenerate
`docs/workflow_yaml_reference.md`), or derive from the model folder's
`config.json` `max_position_embeddings` (`ovat/core/model_scout.py`). NPU:
`ovms_max_prompt_len`. Unknown = hide the meter, never show 0.

## Step 3. Saved sessions

R2 3.4. Backend:
- Move `list_sessions` / `_session_path` out of `ovat/cli/chat_screen.py`
  (a textual module) into a new textual-free `ovat/agentic/sessions.py`; the
  launcher imports it from there (one implementation).
- `AgentSession.save(name)`, `load(name)`, `list_saved()`, `delete(name)`,
  `rename(old, new)`, autosave after each `TurnDone`. Files in
  `.ovat/sessions/` (agent sessions in a subfolder, e.g.
  `.ovat/sessions/agent/`, so the launcher's rag_chat sessions are not mixed).
- Native engine only for load/rewind/fork. Framework engines: save works
  (transcript), resume says plainly "resume needs the native engine".

Server methods to implement (exact models in `protocol.py`; who calls them in
`reference/R5-rpc-method-table.txt`): `session/list` (real list now),
`session/resume` (any id), `session/continue` (most recent),
`session/rename`, `session/delete`, `session/fork`. On load, rebuild the
Projector's history from stored messages (user / assistant with tool_calls /
tool with tool_call_id) and let the client re-read (a fresh `session/read`).

CLI: `ovat agent --continue` and `ovat agent --resume <id>` pass through to
`start_ovat_app_server(continue_session=..., resume_session_id=...)`.

Unhide in the registry filter: `/resume`, `/sessions` (or whatever Vibe calls
them; see `reference/R1-vibe-inventory.md` section 2 for the full slash
command list and section 1.8 for session management).

## Step 4. Rewind and fork

Native: truncate `Session.messages` at a USER message boundary (never cut a
tool-call pair in half; OVMS rejects it). Fork = truncate + save under a new
name. Methods: the `rewind` group (`session/rewind`, its read method; exact
names in R5) and `session/fork`. Vibe's rewind UI picks a user message; send
back the matching index.

## Step 5. Queue edit and steer

Queue add/read/replace/remove/resume already work (Phase 2a). Add
`turn/steer`: Vibe sends extra guidance while a turn runs. OVAT's loop can
take it at the next iteration boundary: put it in a thread-safe slot that
`AgentLoop` reads before its next LLM call and appends as a user message.
Native only; other engines answer "steer needs the native engine" (as an
RPC error with a readable message, not a crash).

## Step 6. Compact

R2 3.7. `AgentSession.compact(instructions)`: one extra `llm.chat` with a
summarize prompt over the history, replace with `[system, summary]`. Emit
start/end so the projector adds a `PublicCheckpointEntry(kind="compaction")`.
Landmine: the summary call is bounded by `model.max_tokens` and on NPU by
the static prompt cap; do not cut a tool-call pair. Manual `/compact` first;
auto-compact only once Step 2's context size exists.

## Step 7. Model and thinking pickers, retry

- Model: R2 3.5. `AgentSession.start(engine, model=...)` with the same
  `model_copy` override the engine switch uses. List choices: genai = local
  folders via `ovat/core/model_scout.find_models`; OVMS = only what the server
  serves (verify `/v3/models` exists on OVMS 2026.4.1 before relying on it;
  otherwise list the workflow's model only). Method: `config/model/write`
  (applies to this session only in Phase 3; writing workflow.yml is Phase 4).
- Thinking: the `enable_thinking` toggle. Off by default. Note R2 3.2: on
  OVMS 2026.2.1 `enable_thinking: false` broke native and openai-agents;
  check on 2026.4.1 before exposing it, hide it if still broken.
- Retry: re-send the last user message (drop the last assistant turn first,
  native only).

## Done when

- Both suites green (3.13 and 3.10), contract tests for every new method,
  pilot tests for every unhidden command, screenshots looked at.
- Real-model rig run: streaming visible, `/resume` of a saved session shows
  the old transcript, `--continue` works, rewind then ask again works.
- `git log --format=%B | grep -i claude` is empty for your commits.
