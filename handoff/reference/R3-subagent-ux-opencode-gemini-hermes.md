# Subagent and A2A UX in other agent TUIs: research for OVAT

Date: 2026-10-16. Sources are shallow clones, read only; nothing was run.

| Project | Path | Commit |
|---|---|---|
| opencode (sst/opencode, no redirect needed) | `scratchpad/refs/opencode` | `32b41bab` (2026-10-16) |
| Gemini CLI | `scratchpad/refs/gemini-cli` | `b25669d8` (2026-10-16) |
| Hermes Agent | `scratchpad/refs/hermes-agent` | `607a0b0d` (2026-10-16) |
| Mistral Vibe (existing clone) | `scratchpad/vibe-ref` | `4ae5c595` (2026-10-09) |

`scratchpad` = `/tmp/claude-0/-home-user/6831d950-f8c2-52ec-b022-39125d79c2be/scratchpad`. All paths below are relative to each repo root.

## Summary

| | opencode | Gemini CLI | Hermes | Vibe |
|---|---|---|---|---|
| License | MIT | Apache-2.0 | MIT | Apache-2.0 |
| Language / TUI stack | TS, Solid + opentui | TS, React + Ink | Python core, TS Ink TUI (`ui-tui`) talking to a Python gateway over JSON-RPC | Python, Textual |
| Subagent definition | `opencode.json` `agent{}` or `.opencode/agents/*.md` (`mode: subagent`) | `.gemini/agents/*.md` (`kind: local|remote`) | Not named. `delegate_task` takes ad-hoc goals; limits are set in `config.yaml` `delegation.*` | TOML profiles (`agent_type = "subagent"`) in user and project agents dirs |
| Invocation tool | `subagent` (agent, description, prompt, sessionID, background, model) | `invoke_agent` (agent_name, prompt) | `delegate_task` (tasks[], action spawn/list/steer/stop) | `task` (agent, task); the unified harness also has `subagent.spawn/list/wait/send_message/interrupt/stop` |
| Running-subagent display | Inline tool row, a child session that you navigate into, a "N subagents" footer, and a composer "Subagents" tab | Bordered group block with one row per agent and the last activity; ctrl+o expands it to the full activity log | Dock above the prompt, plus a full-screen `/agents` tree overlay with tail, steer and kill | List under the input (Main conversation + children), switching to a read-only transcript view |
| A2A | No (0 hits) | Yes: client (`@a2a-js/sdk`) and a separate server package | Yes: client tools and an inbound server, both stdlib with no a2a-sdk, protocol v1.0 | No (0 hits) |
| Editor protocol | ACP (`@agentclientprotocol/sdk`) | ACP (`packages/cli/src/acp`) | ACP (`acp_adapter/`) | ACP (`vibe/acp`) |

---

## 1. opencode (sst/opencode)

### 1.1 License
`LICENSE` is MIT, "Copyright (c) 2025 opencode". Copying code requires keeping the copyright and permission notice in all copies or substantial portions. Nothing else is required. The code is TypeScript (Solid/opentui), so in practice we would port ideas rather than lift code.

### 1.2 Subagents
**Definition**
- Docs: `packages/web/src/content/docs/agents.mdx`. Schema: `packages/schema/src/config/agent.ts`, `packages/core/src/agent.ts`.
- Agents are either JSON under `agent{}` in `opencode.json` or markdown with frontmatter in `~/.config/opencode/agents/` or `.opencode/agents/`. The filename becomes the agent name.
- `mode` is `primary`, `subagent` or `all`.
- Other fields: `description` (required), `model`, `prompt`, `temperature`, `steps`, `hidden` (hidden from `@` autocomplete), `color`, and a per-agent `permission` map.
- `permission.task` is a glob map (`"*": deny, "orchestrator-*": allow, "code-reviewer": ask`) that decides which subagents an agent may spawn. A denied subagent is removed from the tool description entirely.
- Built-ins: primary `build` and `plan`; subagents `general`, `explore` and `scout`; hidden system agents `compaction`, `title` and `summary`.

**Invocation**
- The tool is `packages/core/src/tool/plugin/subagent.ts` (tool name `subagent`; the UI still accepts the legacy `task`).
- Input: `agent`, `description` (a 3-5 word label shown to the user), `prompt`, an optional `model` (`provider/model#variant`), an optional `sessionID` to continue an earlier child, and an optional `background`.
- Every spawn creates a real **child session** (`parentID` = caller), titled with the description.
- Nesting is limited by `experimental.subagent_depth` (default 1). Primary-only agents are rejected.
- A `permission.assert({action: "subagent", resources:[agent.id]})` call runs before spawning, so spawning a subagent can itself be an "ask" approval.
- The list of available subagents is appended to the tool description on each context, compaction or generate hook, filtered by permissions.
- Users can also run a subagent directly with `@agent ...` in the prompt.

**Background and jobs**
- `packages/core/src/session/subagent-job.ts` and `subagent-completion.ts` handle this.
- A foreground subagent blocks the parent tool call (`jobs.block`). The user can press `session.background` (**ctrl+b**, "Background blocking session tools") to detach it; the tool then returns `status:"running"`.
- On completion a **synthetic message** is injected into the parent: `<subagent sessionID=… state=completed|error|cancelled description=…>text</subagent>`, with metadata `{source:"subagent", childID, agent, state}`.
- The model is told never to poll or sleep.

**TUI display** (`packages/tui/src/routes/session/index.tsx`)
- `function Subagent` (around line 3118) renders the tool call as an inline row.
  - The icon is a spinner or `│` while running, `✓` when done, `↳` for a continuation.
  - Label format: `"<Agent> Subagent, <description> | <model>"`.
  - A "Background" badge appears for backgrounded subagents.
  - The row is **clickable** and navigates into the child session.
- `SessionNoticeMessageV2` (around line 1963) renders the completion notice: `↳ Explore finished | <description>`.
  - Colours: error is red, cancelled is the warning colour, otherwise info.
  - The notice is clickable and opens the child.
- Footer (`packages/tui/src/feature-plugins/prompt/footer.tsx`): `"<↓> 2 subagents | 1 shell"`, where the count only includes running sessions in the family. Clicking it opens the picker.

**Subagent picker**
- Composer tab "Subagents": `packages/tui/src/routes/session/composer/subagents-tab.tsx`.
- Shows a tree of the session family, with tree prefixes, as `Agent: title` plus `Running`.
- Keys:
  - `ctrl+a` toggles between active and inactive subagents.
  - Up and down move the selection.
  - Enter navigates into the child.
  - `composer.subagent.interrupt` interrupts the selected child.
  - Pressing Up at the top closes the picker.
- Inside a child session the picker opens by default.

**Navigation keybinds** (`packages/tui/src/config/keybind.ts`)

| Command | Key | Action |
|---|---|---|
| `session.child.first` | Down | Toggle subagent picker |
| `session.child.next` | Right | Next child session |
| `session.child.previous` | Left | Previous child session |
| `session.parent` | Up | Go to parent |
| `agent.cycle` | shift+tab | Cycle primary agents (the docs still say Tab) |
| `agent.list` | `<leader>a` | Agent list dialog (`component/dialog-agent.tsx`) |
| `session.interrupt` | Esc | Interrupt the current session |

**"Mini" mode** (`packages/tui/src/mini/footer.subagent.tsx`, `stream-v2.subagent.ts`)
- Shows a per-child footer tab with status icons: `◔` running, `●` done, `◍` error, `○` cancelled.
- Includes a 14-row inspector transcript and per-child interrupt.
- Children are discovered from tool metadata, the session list by `parentID`, and live events.

**Approvals**
- Permission requests from **all descendant sessions are bubbled up and shown in the root session's view** (`routes/session/index.tsx` around line 197: `permissions = [root, ...descendants].flatMap(permission.list)`). Forms and questions are bubbled the same way.
- The prompt is disabled while any are pending.
- The permission dialog (`routes/session/permission.tsx`) has a dedicated rendering for `subagent`/`task` requests.

**Cancellation**
- Interrupting the parent's tool call interrupts the child (`Effect.onInterrupt(() => sessions.interrupt(child.id); jobs.cancel(child.id))`).
- Per-child interrupt is available from the picker. Results report cancelled or failed with the sessionID so the model can continue the child.

### 1.3 A2A
**Not used.** `grep -rniwE 'a2a|agent2agent'` over the repo, excluding node_modules, finds 0 text hits, and there are 0 hits for `agent-card|well-known/agent`.

What it uses instead:
- **ACP** (`@agentclientprotocol/sdk` 1.6.0 in `packages/cli/package.json`; code in `packages/cli/src/acp/*`). `acp/child.ts` defines a custom `opencode/session/child_update` notification carrying `rootSessionId/childSessionId/parentSessionId/depth` and status `created|running|completed|failed|interrupted`. This projects subagent sessions to editors.
- **MCP** for tools (`packages/core/src/tool/mcp.ts`).
- An HTTP server and SDK of its own (`packages/server`, `packages/sdk`).

---

## 2. Gemini CLI (google-gemini/gemini-cli)

### 2.1 License
`LICENSE` is Apache-2.0. There is no NOTICE file at the root. Source files carry `@license Copyright 2025/2026 Google LLC SPDX-License-Identifier: Apache-2.0` headers.

Copying requires:
- including the Apache-2.0 license text;
- keeping the copyright and SPDX headers;
- adding prominent notices on modified files;
- carrying over a NOTICE file if one exists.

The license includes an express patent grant, which terminates if you bring a patent suit. The code is TypeScript, so in practice we would port designs.

### 2.2 Subagents
**Definition**
- Docs: `docs/core/subagents.md`; loader: `packages/core/src/agents/agentLoader.ts`; registry: `registry.ts`.
- Agents are markdown files with YAML frontmatter in `.gemini/agents/*.md` (project) or `~/.gemini/agents/*.md` (user). The body is the system prompt.
- Fields: `name`, `description`, `kind: local|remote`, `tools` (wildcards `*`, `mcp_*`, `mcp_<server>_*`), inline `mcpServers` isolated per agent, `model`, `temperature`, `max_turns` (30), and `timeout_mins` (10).
- Settings overrides live in `agents.overrides.<name>` (`enabled`, `modelConfig`, `runConfig`).
- Built-ins: `codebase_investigator`, `cli_help`, `generalist`, and `browser_agent` (off by default; it shows a first-run consent dialog).
- Subagents cannot call subagents, even with `*`.

**Invocation**
- A single unified tool, `invoke_agent(agent_name, prompt)`, in `packages/core/src/agents/agent-tool.ts` (`AGENT_TOOL_NAME` in `tools/tool-names.ts:191`).
- It dispatches to `LocalSubagentInvocation`, `RemoteAgentInvocation` (A2A), `BrowserAgentInvocation`, or the session-based variants (`local-session-invocation.ts`, `remote-session-invocation.ts`).
- The policy engine treats `agent_name` as a virtual tool name, so `[[rule]] toolName="codebase_investigator" decision="deny"` works (`docs/reference/policy-engine.md` "Special syntax for subagents").
- Users can force an agent with `@agent_name ...`, which injects a system note.
- Parallel subagents come from the model issuing several `invoke_agent` calls in one turn.

**TUI display** (`packages/cli/src/ui/components/messages/`)
- `SubagentGroupDisplay.tsx` renders a rounded-border block with a header.
  - Header text: "Running Agent...", "Running 3 Agents...", "3 Agents (1 running, 2 completed)...", or "3 Agents Completed".
  - It shows `(ctrl+o to expand)`, registered with a global overflow system so the hint appears in the sticky footer.
  - **Collapsed view:** one row per agent, `icon agentName | <last activity> <args>`. The icon is `!` running, `✓` done, `ℹ` cancelled or `✗` error. A thought is prefixed with `💭`. "Finished Early (MAX_TURNS)" appears when the terminate reason is not GOAL.
  - **Expanded view:** `SubagentProgressDisplay.tsx`, the full activity list.
    - Each tool call gets a spinner, ✓, ✗, or a strikethrough when cancelled.
    - Args are summarised by `formatToolArgs`, which picks `description`, `command`, `file_path`, `query` or `url` and truncates to 60 characters.
    - The final result is rendered as Markdown.
- `SubagentHistoryMessage.tsx` shows the `🤖 <agent> Trace (N items)` history item.
- **Important filter:** in `packages/cli/src/ui/hooks/useToolScheduler.ts` around lines 124-175, each subagent has its own `schedulerId`.
  - For non-root schedulers the main UI **only shows tool calls that are awaiting approval**, or that were already shown.
  - Subagent reads and searches therefore never flicker into the main transcript, but an approval request from a subagent appears in place, attributed to that subagent.
  - Separately, a `SUBAGENT_ACTIVITY` message-bus channel feeds the activity history.
- `/agents list|enable|disable|config|reload` (`ui/commands/agentsCommand.ts`).
  - `list` renders `views/AgentsStatus.tsx` with "Local Agents" and "Remote Agents" sections, showing displayName (name) and a Markdown description.
  - `config` opens `AgentConfigDialog.tsx`, a form with Enabled, Model, Temperature, Top P/K, Max Output Tokens, Max Time and Max Turns.
- **Trust for project agents:** `NewAgentsNotification.tsx` together with `core/src/agents/acknowledgedAgents.ts`.
  - The first time project-level agents (or changed ones, compared by content hash) are seen, a warning-bordered "New Agents Discovered" box lists them.
  - The choices are "Acknowledge and Enable" or "Do not enable (Ask again next time)".
  - Acknowledgements are stored per project path, agent name and hash.

**Approvals**
- Default policy `packages/core/src/policy/policies/agents.toml` **allows** `invoke_agent` ("Subagents are trusted to handle their own confirmations").
- Each subagent's tools are cloned onto a derived message bus, `parentMessageBus.derive(definition.name)` (`local-executor.ts:175-215`), so confirmations carry the subagent name.
- Policy rules can be scoped with `subagent = "pr-creator"`.
- **Remote agents get a dynamic `ASK_USER` rule** (`registry.ts:404-430`) unless the user wrote one. The confirmation is of type `info`: "Call Remote Agent: <name>" / `Calling remote agent: "<query>"` (`remote-invocation.ts`).
- In Plan mode only allow-listed subagents may run (`policies/plan.toml`).

**Cancellation**
- An `AbortSignal` is passed through. Cancelled tool calls are rendered with strikethrough and the header becomes "Agent Cancelled".

### 2.3 A2A (yes, both client and server)

**Client** (`packages/core/src/agents/`, dependency `@a2a-js/sdk` 0.3.11)
- **Configuration:** `kind: remote` markdown with `agent_card_url` or inline `agent_card_json`. Several remote agents can share one file as a YAML list. Docs: `docs/core/remote-agents.md`.
- **Discovery:** `a2a-client-manager.ts` `loadAgent()`.
  - The card is fetched **without auth first, then retried with auth on 401/403**.
  - Transports are REST, JSON-RPC and gRPC, chosen from the card's interfaces.
  - `normalizeAgentCard` handles proto alias names.
- **Description:** `registry.ts:550-575` merges the user description, the card description and the card `skills` ("Skills:\n name: desc") into what the model sees.
- **Auth:** `auth-provider/` covers `apiKey`, `http` (Bearer, Basic or a raw scheme), `google-credentials` (ADC, restricted to `*.googleapis.com` and `*.run.app`) and `oauth` (auth code with PKCE).
  - OAuth endpoints and scopes are discovered from the card's `securitySchemes`.
  - Secret values may be `$ENV`, `!command`, or a literal.
  - Requests are retried on 401/403 up to 2 times.
  - Auth config is validated against the card's `securitySchemes`, and a warning `[agent] Agent requires authentication: …` is shown while the agent stays registered.
- **Streaming:** `remote-invocation.ts` `sendMessageStream`.
  - Each chunk is fed to `A2AResultReassembler` (`a2aUtils.ts`), which handles `status-update`, `artifact-update` (with `append`), `task` and `message`.
  - It renders `SubagentProgress {state, recentActivity, result}` live through `updateOutput`, using the same `SubagentGroupDisplay` as local agents.
  - When a task's status is `auth-required` it adds an instruction message.
  - Artifacts are rendered as `Artifact (<name>):\n…`.
  - It falls back to the last agent message in `history` if a terminal task carries no text.
- **Multi-turn:** `contextId` and `taskId` are kept per agent name in a static `sessionState` map, so the next call continues the same remote task. `taskId` is cleared on a terminal state (`extractIdsFromResponse`).
- **Cancellation:** handled by aborting the local stream. `A2AClientManager.cancelTask()` exists (`a2a-client-manager.ts:278`) but **nothing calls it**, so `tasks/cancel` is never sent.
- **Errors:** typed user-facing messages in `a2a-errors.ts`.

**Server** (`packages/a2a-server`, marked "experimental"; express plus `@a2a-js/sdk/server`)
- Agent card "Gemini SDLC Agent" in `src/http/app.ts:45`:
  - streaming true, `stateTransitionHistory` true;
  - bearer or basic security schemes;
  - skill `code_generation`;
  - served at `/.well-known/agent-card.json`.
- The executor (`src/agent/executor.ts`, `task.ts`) maps a CLI turn to an A2A task.
  - The state becomes `input-required` at the end of each turn.
  - `tasks/cancel` aborts the active loops (`executor.ts:279`) and sets the state to `canceled`.
- **Tool approvals over A2A:** a tool in `awaiting_approval` publishes a status-update whose `metadata.coderAgent.kind = "tool-call-confirmation"` (`types.ts` `CoderAgentEvent`).
  - The client answers with a DataPart `{callId, outcome: proceed_once|cancel|proceed_always|…}` (`task.ts:977 _handleToolConfirmationPart`).
  - Other event kinds: `tool-call-update`, `text-content`, `state-change`, `thought`, `citation`, and `agent-settings` (`autoExecute`, `workspacePath`).
- Extra non-A2A endpoints: `POST /tasks`, `/executeCommand`, `GET /listCommands`, `/tasks/metadata`. Task persistence uses a GCS store (`persistence/gcs.ts`) or in-memory.

**Other protocols:** ACP for editors (`packages/cli/src/acp/*`) and MCP.

---

## 3. Hermes Agent (NousResearch/hermes-agent)

### 3.1 License
`LICENSE` is MIT, "Copyright (c) 2025 Nous Research". Copying requires keeping the copyright and permission notice. The core and the A2A plugin are **Python**, so they are the most directly reusable for OVAT. The TUI is TS/Ink.

### 3.2 Subagents
**Definition**
- There are **no named agent definitions**. `delegate_task` (`tools/delegate_tool.py:696`) takes `tasks[]`, each with `goal`, `context`, an optional `output_schema` (JSON Schema validated by the parent, with one correction retry), `images` and `group`.
- `config.yaml` `delegation.*` sets the limits: `max_concurrent_children`, `max_spawn_depth`, the orchestrator, `independent_completions`, `subagent_auto_approve`, and provider/model pins.
- Children can also be external ACP agents through `acp_command`/`acp_args`, which are trusted config only and stripped from model input (`_MODEL_HIDDEN_TASK_FIELDS`).

**Invocation and control**
- The same tool has `action: spawn|list|steer|stop` plus `subagent_id` and `message`.
- `steer` queues course-correction text, which is appended to the child's next tool result.
- Top-level delegations **always run in the background**: dispatch returns transcript paths, and results come back as a new message between turns, either per task or per `group`.
- The tool description tells the model that child summaries are self-reports and must be verified, and that it should end its turn instead of polling.
- Further files: `tools/delegate_tool_dispatch.py`, `async_delegation.py`, `delegation_live_log.py`, `delegate_tool_progress.py`.

**TUI** (`ui-tui/src/`)
- **Live dock** above the prompt (`components/agentsPanel.tsx` `LiveAgentsPanel`).
  - Header: `▾ N live agents | N procs`, with a collapsed one-line form `▸ … | <first activity>`.
  - Each row shows a status glyph, the goal, the elapsed time, and a second line `↳ <current activity>`.
  - The dock row budget is split between agents and background processes.
- **`/agents` (alias `/tasks`) full-screen overlay**: `components/agentsOverlay.tsx`, "spawn-tree dashboard (live audit + kill/pause controls)", described in the table below.
- `components/agentControls.tsx`: `AgentLiveTail` polls the `subagent.tail` RPC every 1.5s, and `AgentSteerForm` sends `subagent.steer`.
- The inline transcript section for subagents is in `components/thinking.tsx`.

| Overlay feature | Detail |
|---|---|
| Layout | Tree (`lib/subagentTree.ts`) with depth, a duration and token totals header, a model mix, "hotness" sparklines, and a timeline when height is at least 32 |
| `j/k`, arrows, `g/G` | Move the cursor |
| Enter | Live **tail** |
| `l` / right | Detail |
| `s` | Cycle sort: depth-first, tools-desc, duration-desc, status |
| `f` | Cycle filter: all, running, failed, leaf |
| `e` | **Steer** the selected child (text input) |
| `x` / `X` | Kill one / kill the subtree |
| `p` | Pause new spawns (`delegation.pause`) |
| `[` / `]` | Step through **past spawn trees**; they are saved by `spawn_tree.save/list` RPCs in `tui_gateway/methods_session.py` and can be replayed |
| Esc, `q`, ctrl+t | Close |
| `/agents pause|resume|status` | Non-interactive forms |

**Approvals and cancellation**
- Children run non-interactively. Dangerous-command approvals are **auto-denied by default**, or auto-approved and logged if `delegation.subagent_auto_approve` is set (`tools/delegate_tool_config.py:40-60`).
- Children cannot use `clarify`, memory or cron.
- `/stop`, `/new` or process exit halts the whole tree; each child returns "interrupted" with partial output.

### 3.3 A2A (yes, client and server; stdlib only, no a2a-sdk)
Code is in `plugins/platforms/a2a/`: `README.md`, `DESIGN.md`, `tools.py`, `adapter.py`, `protocol.py`, `security.py`. It targets **A2A v1.0** JSON-RPC (states `TASK_STATE_*`) and tolerates 0.3 cards and parts.

**Client: five model tools**

| Tool | What it does |
|---|---|
| `a2a_discover(url)` | Prints the card summary: name, description, RPC URL, protocol, streaming, push, auth, and up to 20 skills |
| `a2a_call(agent, message, context_id?)` | Sends a blocking `SendMessage` (no streaming on the client) |
| `a2a_list()` | Lists configured peers, saved conversations and metrics |
| `a2a_history(context_id)` | Recalls a saved conversation |
| `a2a_orchestrate(capability, message, mode=all|first|best)` | Fans out to every peer whose `capabilities` matches |

- `a2a_call` output is `"[agent | context <id> | <state>]\n<reply>"`. On `input-required` it adds an instruction to call again with the same context_id.
- Peers are configured under `a2a_agents:` in `config.yaml` with `url`, `auth: {type: bearer, token}`, `timeout` and `capabilities`; a direct URL also works.
- Outbound text has credential-shaped strings redacted.
- There is **no TUI-specific rendering**: A2A calls show as ordinary tool calls.

**Server: an inbound "platform adapter" like Telegram or Slack**
- Built on stdlib `http.server`.
- The card is served at `/.well-known/agent-card.json` (legacy `agent.json` also answers). Its skills are built **dynamically from the live tool registry**.
- Methods: `message/send`, `message/stream` (SSE), `tasks/get|list|cancel|subscribe`, and push notification CRUD.
- Inbound tasks are injected into the **live** agent session keyed by `contextId`.
- `[INPUT_REQUIRED]` reply prefix maps to `TASK_STATE_INPUT_REQUIRED`.
- `tasks/cancel` marks the task canceled but cannot abort the in-flight turn (documented limitation).

**Security** (`security.py`)
- With no token, it binds 127.0.0.1 only. Widening the bind needs a token **and** `A2A_HOST`.
- Per-peer tokens (`A2A_PEER_TOKENS=name:tok`) give each peer an authenticated identity, used for the rate limit, trust gate (`A2A_TRUSTED_PEERS`) and audit.
- Inbound text passes a prompt-injection filter and is framed as untrusted. Remote peers cannot run slash commands.
- An anti-loop ping-pong cap applies (`A2A_MAX_PINGPONG_TURNS`=5).
- Push callbacks are HMAC-signed and SSRF-guarded.
- Every exchange goes to an audit log, `~/.hermes/a2a_audit.jsonl`. Conversations persist outside compaction.

**Other protocols:** ACP adapter for editors (`acp_adapter/`), ACP child agents for delegation, MCP, and an MCP server (`mcp_serve.py`).

---

## 4. Mistral Vibe (existing port source)

### 4.1 License
`LICENSE` is Apache-2.0, "Copyright 2025 Mistral AI", with no NOTICE file. The obligations match Gemini CLI: keep the license and copyright, and mark modified files. OVAT is already porting from it, so this applies to the port as a whole.

### 4.2 Subagents
**Definition**
- `vibe/core/agents/models.py` defines `AgentProfile` TOML files (`display_name`, `description`, `safety`, `agent_type = "agent"|"subagent"`, plus overrides such as `enabled_tools` and `system_prompt_id`).
- Profiles are found in user agents dirs (`GLOBAL_AGENTS_DIR`) and project agents dirs (`vibe/core/config/harness_files/_harness_manager.py:166-200`).
- The only built-in subagent is `explore` (read-only: grep, read_file, skill).
- Primary agents act as approval modes: ask, plan, accept-edits, smart-approve, auto-approve.

**Invocation**
- **Legacy `task` tool** (`vibe/core/tools/builtins/task.py`, args `TaskArgs{task, agent="explore"}` in `vibe/core/subagents.py`).
  - Permission defaults to ASK, with an `allowlist` (default `["explore"]`, which skips the ask) and a `denylist` of fnmatch patterns.
  - Depth limit is 1. Only `agent_type=subagent` profiles may be used.
  - The runner (`vibe/app_server/_sessions.py:317`, `SessionRuntimeRegistry.run`) creates a persisted child session linked to the parent tool call (`link_subagent`). It streams `ToolStreamEvent("task", "<tool>: <result text>")` per child tool result.
  - It returns `TaskResult{response, turns_used, completed}`. The display reads "Running explore agent: …" and then "Completed in N turns" or "Interrupted after N turns".
  - Cancelling the parent generator interrupts the child turn.
- **Unified harness:** `subagent.list|spawn|wait|send_message|interrupt|stop` operations (`vibe/app_server/_unified_harness_backend_adapter.py:8789`), with a `SubagentEffectDetail{input, child_session_id, agent_name}` card (`_effect_models.py:288`).
- Toggles: `enable_subagents` and `show_subagent_status_list` (`vibe/core/config/vibe_schema.py:617-648`).

**TUI**
- `vibe/cli/textual_ui/widgets/chat_input/subagent_list.py` (`SubagentList`, a `NavigableOptionList` under the input):
  - The first row is "Main conversation", followed by `<AgentType> (<name>) [running|blocked|failed|stopped|ready] | 12k tokens`.
  - Status colours: success = running, warning = blocked ("Waiting for input"), error = failed.
  - It keeps the "active batch" (running or blocked children and siblings spawned with them) visible.
  - Pressing Down from the input focuses the list (`ChatTextArea.NavigateBelow`); pressing Up at the top returns to the input.
  - While viewing a child, the border title reads "Viewing X | read-only | Esc to return".
- `vibe/cli/textual_ui/widgets/subagent_transcripts.py` (`SubagentTranscripts`):
  - An LRU of 5 lazily mounted child transcript trees, built with the same `build_history_widgets` as the main chat.
  - The parent's instruction is pinned as the first message, and the tail is merged incrementally.
  - It shows "This subagent is ready for a new instruction…" on the transition to idle, and "This subagent is closed…" on error.
- `vibe/cli/textual_ui/app.py:1370-1560`:
  - `_show_subagent_chat` hides the main messages area and the input box, focuses the list, and restores per-transcript scroll offsets.
  - The context-progress meter switches to the child's tokens.
  - Typing while in a child view shows "You can't interact with a subagent directly. Return to Main conversation and ask the main agent to stop it."

**Approvals:** all callbacks (approval or question) are serialised through one bottom `ApprovalApp` with a `_pending_callbacks` deque (`app.py:3076-3150`). Children show `[blocked]` in the list while waiting.

### 4.3 A2A
**Not used.** `grep -rniwE 'a2a|agent2agent' vibe/` returns 0. Vibe uses **ACP** (`vibe/acp/*`, the `acp` Python package; subagent effects are projected in `vibe/acp/session_updates.py:643,706`), MCP, and its own app-server protocol (`vibe/app_server`).

---

## 5. UX ideas for OVAT's Textual TUI

OVAT starts from Vibe's `SubagentList` and `SubagentTranscripts`. The ideas below layer the best patterns from the other three on top, and are tailored to OVAT's three kinds of agent: local workflow subagents, cloud subagents, and A2A remote agents.

### 5.1 Listing agents
- **`/agents` with grouped sections**, as in Gemini `AgentsStatus.tsx`: Local, Cloud and Remote (A2A). Each entry shows displayName (name), a description, and for A2A the card's skills, endpoint host, auth type and a reachability or auth status (OK, auth required, unreachable). Gemini merges user description, card description and skills (`registry.ts:550`); do the same for the model-facing description.
- **Subcommands** `/agents list|enable|disable|config|reload`, as in Gemini `agentsCommand.ts`, plus `/agents discover <url>`, which renders an agent card like Hermes `a2a_discover`.
- **Trust prompt for project-level agent files and new A2A endpoints**, as in Gemini `NewAgentsNotification.tsx` and `acknowledgedAgents.ts`. Use a hash-pinned acknowledgement per project, and ask again when the file or card changes.
- An **agent picker dialog** (`<leader>a`, opencode `dialog-agent.tsx`) and **shift+tab cycling of primary agents**. Vibe already does this with its approval-mode agents.

### 5.2 Delegating
- **One delegate tool** with a `kind` resolved from the registry, so local, cloud and remote agents share the same rendering path (Gemini `invoke_agent`, `agent-tool.ts`).
- Include a short `description` label (3-5 words, shown to the user) and an optional `session_id`/`context_id` to continue a child (opencode `subagent.ts` input schema).
- **`@agent` mention** in the composer to force a specific agent (opencode and Gemini).
- **Permission globs per agent**, as in opencode `permission.task` (`"*": deny, "reviewer": ask`). Remove denied agents from the tool description entirely.
- Optional **`output_schema` per task**, validated by the parent (Hermes `delegate_tool.py`). This is useful for workflow subagents that must return structured data.

### 5.3 Watching progress live
- **Inline group block** for each delegate call (Gemini `SubagentGroupDisplay.tsx`).
  - Header: "Running 3 agents (1 running, 2 done)…".
  - One collapsed row per agent: `icon name | last activity <short args>`.
  - One key (ctrl+o) toggles the full activity log.
  - Use `formatToolArgs`-style argument summarising.
- **Dock above the input** listing live agents (Hermes `agentsPanel.tsx`): glyph, goal, elapsed time, and a second line `↳ current activity`. A footer counter, "↓ 2 subagents", acts as a click or keyboard target (opencode `feature-plugins/prompt/footer.tsx`). Vibe's `SubagentList` already sits in this place; add elapsed time and last activity to its rows.
- **Don't flood the main transcript with child tool calls.** Show a child's tool call in the parent view only when it needs approval (Gemini `useToolScheduler.ts:146-158`).
- **For A2A, map protocol events onto the same progress model** (Gemini `a2aUtils.ts` `A2AResultReassembler`):
  - `status-update.message` becomes activity lines;
  - `artifact-update` (with append) becomes named artifact sections;
  - `auth-required` and `input-required` become a highlighted "needs action" state;
  - a final task without text falls back to the history.
- Show the remote `taskId` and `contextId` and the A2A task state (submitted, working, input-required, completed, canceled, failed, rejected) in the row.

### 5.4 Opening a subagent's transcript
- **Keep Vibe's model:** Down from the input goes to the list, Enter opens a read-only child transcript, and Esc returns. Keep the LRU of mounted transcripts, the pinned parent instruction, and per-view scroll memory.
- **Add opencode's navigation keys:** Right and Left cycle sibling children, Up goes to the parent. Make the inline delegate row and the completion notice **clickable** so they open the child (opencode `routes/session/index.tsx` `Subagent` and `SessionNoticeMessageV2`).
- **For A2A agents the "transcript" is the task history:** status messages, artifacts and the `history` array. Render it with the same widget.
- **Steer instead of read-only.** Vibe tells the user to "ask the main agent". Hermes lets the user steer a running child directly (`agentControls.tsx` `AgentSteerForm`, `e` key) by queueing text into the child's next tool result. For A2A agents, steering maps naturally onto a follow-up `message/send` with the same `contextId`/`taskId`.

### 5.5 Approvals for remote and cloud calls
- **Ask by default before the first call to a remote or cloud agent** (Gemini `registry.ts:420-430` dynamic `ASK_USER`, plus the `info` confirmation "Call Remote Agent: X / Calling remote agent: "<query>""). Local subagents are allowed by default.
  - Offer "allow once", "always for this agent", and "always for this agent in this project".
  - Show the endpoint host and the exact outgoing text (after redaction, as Hermes `security.redact_outbound` does).
- **Bubble every descendant's approval into the root view's approval bar, labelled with the agent name**:
  - opencode `routes/session/index.tsx:197-215`;
  - Gemini `messageBus.derive(subagentName)` in `local-executor.ts:175`;
  - Vibe already serialises callbacks through one `ApprovalApp` deque.

  Mark the child `[blocked]` in the list and fire the terminal notification.
- **Policy rules scoped by agent** (Gemini `subagent = "name"` in policy TOML) and a safe default for headless or background children: deny, or auto-approve plus log (Hermes `delegation.subagent_auto_approve`).
- When a remote task returns **`auth-required`**, show a dedicated prompt with a "re-authenticate" action. Gemini only appends a text instruction.
- Validate the auth config against the card's `securitySchemes` when the agent loads, and show a warning badge (Gemini `registry.ts:526-547`).

### 5.6 Parallel subagents
- **Group parallel calls in one block** with aggregate counts (Gemini). Use a **tree view** for nested spawns (Hermes `lib/subagentTree.ts`, opencode `sessionFamily` prefixes).
- **A dashboard overlay** (`/agents` or `/tasks`, Hermes `agentsOverlay.tsx`):
  - sort (status, duration, tool count);
  - filter (running, failed, leaf);
  - kill one with `x` or the subtree with `X`;
  - pause new spawns with `p`;
  - live tail with Enter;
  - token and duration totals.

  In Textual this is a `ModalScreen` with a `Tree` or `DataTable`.
- **Background running:**
  - ctrl+b detaches a blocking foreground subagent (opencode `session.background`);
  - completion arrives as a structured notice that names the child (opencode `subagent-completion.ts`);
  - the model is told not to poll or sleep (opencode and Hermes tool descriptions).
- **Fan-out to remote peers by capability**, with modes all, first and best (Hermes `a2a_orchestrate`). This could appear as one group block with a row per peer.
- **Concurrency and depth limits shown in the UI** (Hermes `/agents status`: max depth, max children, paused state).

### 5.7 Serving OVAT over A2A
- **Card:**
  - Serve `/.well-known/agent-card.json`, and answer the legacy `/.well-known/agent.json` too.
  - Build skills **dynamically from the enabled tools and workflows** (Hermes `protocol.build_agent_card`), with an allow-list setting (`A2A_ADVERTISED_TOOLSETS`).
  - Set `capabilities.streaming=true` (Gemini `a2a-server/src/http/app.ts:45`).
- **Tool approvals over A2A, following Gemini's executor:**
  - when an OVAT tool needs approval, emit a `status-update` in `input-required` with structured metadata (`{kind: "tool-call-confirmation", callId, tool, args}`);
  - accept a DataPart `{callId, outcome: proceed_once|proceed_always|cancel}` (`a2a-server/src/agent/task.ts:542-570, 977-1015`);
  - support an `autoExecute` setting per client.
- **`tasks/cancel` must really abort the running turn**, as Gemini `executor.ts:279` does with abort controllers. Hermes only marks the task cancelled (a documented gap).
- **Safe defaults** (Hermes `security.py`, `DESIGN.md`):
  - with no token, bind loopback only;
  - per-peer tokens give a named identity that drives rate limits, an allow-list and audit;
  - treat inbound text as untrusted (prompt-injection framing, no slash commands);
  - an anti-loop turn cap per `contextId`;
  - an audit JSONL file;
  - redact outbound secrets.
- **TUI surface for the server:**
  - a status-bar badge, "A2A :9900 | 2 peers | 1 task";
  - an `/a2a` (or `/agents serve`) screen listing inbound tasks with peer identity, state and elapsed time, plus cancel and an "open transcript" action using the same transcript widget;
  - an **inbound approval prompt** that names the peer ("peer `alice` wants to run `shell`").

  Hermes injects inbound tasks into the live session; OVAT can instead show them as child sessions in the same `SubagentList` with a "remote caller" badge, which keeps the main transcript clean.
- **Library choice:**
  - the official Python `a2a-sdk` gives JSON-RPC, REST and gRPC transports and a task store, matching Gemini's use of `@a2a-js/sdk`;
  - Hermes shows a stdlib-only v1.0 JSON-RPC implementation in about 1,800 lines (MIT) if dependencies matter;
  - either way, test against both v0.3 and v1.0 cards. Gemini normalises card aliases (`normalizeAgentCard`), and Hermes accepts `kind`/`type` part shapes.

### 5.8 Top patterns to adopt, by source file

1. Show subagent tool calls in the parent view only when they need approval, and attribute them to the subagent. Source: gemini-cli `packages/cli/src/ui/hooks/useToolScheduler.ts`, `packages/core/src/agents/local-executor.ts` (`messageBus.derive`).
2. A collapsible group block for parallel agents, with one live row each and aggregate counts. Source: gemini-cli `packages/cli/src/ui/components/messages/SubagentGroupDisplay.tsx`.
3. A2A stream reassembly into one progress model covering status messages, artifacts, auth-required and history fallback. Source: gemini-cli `packages/core/src/agents/a2aUtils.ts`.
4. A child-session picker with Down, Left, Right and Up navigation, clickable inline rows and completion notices, and per-child interrupt. Source: opencode `packages/tui/src/routes/session/composer/subagents-tab.tsx`, `routes/session/index.tsx`, `config/keybind.ts`.
5. Bubble every descendant's approvals and questions into the root view. Source: opencode `packages/tui/src/routes/session/index.tsx` (lines 197-215).
6. Ask by default for remote agents, with an "info" confirmation, and a hash-pinned trust prompt for project agents. Source: gemini-cli `packages/core/src/agents/registry.ts`, `remote-invocation.ts`, `packages/cli/src/ui/components/NewAgentsNotification.tsx`.
7. A spawn-tree dashboard with sort, filter, tail, steer, kill and pause, plus a live dock above the input. Source: hermes-agent `ui-tui/src/components/agentsOverlay.tsx`, `agentsPanel.tsx`, `agentControls.tsx`.
8. An A2A server that relays tool approvals as `input-required` plus a DataPart reply, with real cancel and secure defaults. Source: gemini-cli `packages/a2a-server/src/agent/task.ts`, `executor.ts`; hermes-agent `plugins/platforms/a2a/security.py`, `DESIGN.md`.

Runners-up:
- background detach (ctrl+b) with completion notices: opencode `core/src/session/subagent-completion.ts`;
- capability fan-out across peers: Hermes `plugins/platforms/a2a/tools.py` `a2a_orchestrate`;
- auth providers with an unauthenticated-then-authenticated card fetch: Gemini `a2a-client-manager.ts` and `auth-provider/`.
