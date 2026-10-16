# 03. Phase 4: config writes, MCP panel, ask-user, todo, theme, voice, diagnostics

Do Phase 3 first (02). Read 01-ARCHITECTURE.md section 4 (recipe) before
starting. For what Vibe's UI offers in each area see
`reference/R1-vibe-inventory.md` sections 1.9 (settings and pickers), 1.10
(integrations), 1.11 (voice), section 2 (slash commands) and section 4
(screens). For OVAT's backend facts see `reference/R2-ovat-capabilities.md`
3.8 (MCP), 3.9 (image, voice), 3.11 (permission rules).

Every item: backend in textual-free OVAT code with a unit test, server
handler with a contract test, slash command unhidden with a pilot test.

---

## Step 1. Config writes to workflow.yml

Methods: `config/write` and its reload sibling, `config/model/write` to
persist (Phase 3 made it session-only). Exact names and models: grep
`"config/` in `reference/R5-rpc-method-table.txt`.

Rules:
- Validate with OVAT's own `ovat/config/workflow.py` models BEFORE writing.
  An invalid change is refused with the validation message; the file is not
  touched.
- Write atomically (temp file + `os.replace`) and keep comments if possible.
  If the project uses plain `yaml.safe_dump` today, check whether ruamel is
  already a dependency; do NOT add a new dependency without the owner. If it
  cannot keep comments, say so in the confirmation the user sees.
- Only fields OVAT really has. The Vibe settings screen shows about thirty
  Vibe fields; `ovat/tui/server/views.py` already maps them to "off". Writes
  to those fields answer "not used by OVAT" and change nothing.
- After a write that needs a rebuild (model, device, engine), use the same
  path `/engine` uses (`AgentSession.start(...)`, Phase 2b added a switch
  through the handle) so history behaviour is consistent.

## Step 2. MCP panel

R2 3.8. Backend: a public `status()` on `MCPStdioServer` (connected,
error, command, tool names) and an `mcp` list in `describe_agent`. Server:
the `mcp_catalog/*` read methods and the toggle (enable/disable for THIS
session only; persistent toggles go through Step 1's config write). Vibe's
MCP screen then works unchanged. Restart/reconnect is a larger change (one
manager coroutine owns connect and unwind, see AGENTS.md section on MCP);
leave it out unless trivial. PR 81 (fix/mcp-orphan) changed MCP connect
cleanup: merge main into feat/a2a after PR 81 lands, before this step, so you
build on the fixed code.

## Step 3. Ask-user question tool

Vibe has question callbacks (a callback entry of the question kind; see
`models.py` for the question callback models and `approvals.py` for how
OVAT already does approval callbacks). Backend: a builtin tool `ask_user`
(`ovat/tools/`) that blocks the worker thread on a broker like
`ApprovalBroker`, with the question and optional choices. Server: send it as a
question callback; `callback/result` resolves it. Headless (`ovat run`, A2A
server) must not hang: no broker = the tool returns "no person to ask" at
once. Tool off by default; enabled per workflow (`tools: - name: ask_user`).

## Step 4. Todo tool

Vibe renders a todo list from a tool's effect. Add a builtin `todo` tool
(write/read the list in session state, no disk) and map its effect to Vibe's
todo effect detail in `ovat/tui/server/bridge.py`. Small models misuse
planning tools: keep it opt-in per workflow.

## Step 5. Theme

Vibe has a theme picker (`/theme`) that writes Vibe's config. Map it to
OVAT's UI prefs file (the launcher already keeps UI prefs; reuse that file,
do not invent a second). Textual themes only, no network.

## Step 6. Voice (local only)

R2 3.9. Implement Vibe's `VoiceManagerPort` in a textual-free
`ovat/agentic/voice.py`: mic capture needs a new optional dependency (e.g.
`sounddevice`) behind a new extra, NOT in `tui`. Transcribe with OVAT's
existing whisper pipeline (`ovat/tools/transcribe.py`, `_load_pipeline`),
model from the workflow's `transcribe` tool entry or `OVAT_WHISPER_MODEL`.
Load on a worker thread. If the extra or model is missing, the voice key
shows a clear message, nothing crashes. Never send audio anywhere. Ask the
owner before adding the dependency (rule: no new deps without him).

## Step 7. Diagnostics

`diagnostics/list` exists (Phase 2a). Feed it from `ovat doctor`'s checks
(`ovat/tui/widgets/doctor.py` runs them) so Vibe's diagnostics view and
`/doctor` agree. One source of truth.

## Step 8. Permission rules (optional, ask the owner)

R2 3.11. "Approve permanently" currently = SESSION. Persist as
`.ovat/permissions.json` and preload the broker, only where a broker exists
(never auto-allow in headless mode; `gate()`'s "no approver means no" stays).

## Done when

Same bar as Phase 3: both suites green, contract + pilot tests, screenshots
looked at, real-model rig run of each new screen, no Claude attribution.
