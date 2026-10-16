# Mistral Vibe TUI: feature inventory for porting into OVAT

Source: Mistral Vibe clone at `scratchpad/vibe-ref`, commit `4ae5c59` (v2.26.1), Apache-2.0.
Target: `/home/user/wt/feat-a2a/ovat/tui/vibe/` (copied files) and `/home/user/wt/feat-a2a/ovat/tui/app.py` (OVAT's own app).
Paths below are relative to the Vibe repo root (`vibe/...`). "Copied: yes" means the same relative path exists under `ovat/tui/vibe/` (with `vibe.` replaced by `ovat.tui.vibe.`).

Line counts are `wc -l`. `app.py` refers to `vibe/cli/textual_ui/app.py` (6,730 lines, **not** copied; OVAT has only its `ChatScroll` class, 91 lines). "L1234" means a line in that file.

---

## 0. Architecture: read this first

1. **Vibe's TUI never imports `vibe.core`.** Every backend call goes through the "app server" client: `vibe/app_server/session.py` (`AppServerSession`, 1,083 lines) plus `vibe/app_server/resources.py`, which exposes 18 resource objects on `app_server.resources.*`: `account, identity, provider_auth, config, agents, runtime, mcp, plugins, skills, shell, sessions, review, workspace, loops, telemetry, narration, feedback, vibe_code`. These speak JSON-RPC methods such as `session/turn/enqueue`, `config/write`, `mcp_catalog/toggle` and `vibeCode/teleport/start` to an in-process server (`vibe/app_server/server.py`, `_handler.py` and others). The boundary is enforced by a test (`tests/cli/textual_ui/test_app_server_boundary`, mentioned in `vibe/cli/textual_ui/constants.py` L7 and `widgets/skills_browser.py` L6).
   - OVAT already has the **wire models** (`app_server/models.py`, `protocol.py`, `events.py`, `config.py`, `_effect_models.py`, `review.py`, `provider_auth.py` were all copied).
   - OVAT does **not** have the client side: `session.py`, `resources.py`, `_runtime_resources.py` (504 lines), `_session_resources.py` (645), `_integration_resources.py` (834), `_service_resources.py` (141), `client.py` (349), `client_state.py` (137), `connection.py` (136), `transport.py` (189), `host.py` (267), `_streaming.py`, `_turn_input.py`, `client_tools.py` and `telemetry_port.py`.
   - **Porting implication:** the handlers in `app.py` can be kept almost unchanged if OVAT provides an object with the same surface as `AppServerSession` and `resources.*`, backed by `AgentSession` through the bridge. The alternative is to rewrite each handler against `ovat/tui/bridge.py`. Section 7 lists the full RPC surface the TUI uses.
2. **Bottom-panel model.** Every picker, browser or dialog except the config screen and the todo overlay is a "bottom app". `app.py` hides the `ChatInputContainer` and mounts the widget into `#bottom-app-container`. These apps are listed in the `BottomApp` enum (app.py L378): `Approval, ConnectorAuth, Input, LogLevelPicker, MCP, MCPOAuth, ModelPicker, Plugins, ProxySetup, Question, ThemePicker, ThinkingPicker, SkillsBrowser, Rewind, VibeCodeProjectPicker, VibeCodeProjectCreate, SessionPicker, Voice`. The functions `_switch_from_input`, `_replace_bottom_app`, `_switch_to_input_app` and `_try_interrupt_bottom_app_escape` (app.py L5085–5756) handle switching, focus and Escape for all of them.
3. **Experimental vs legacy harness.** Several features exist only when `runtime.experimental_harness` (the "unified harness") is on: `/plugins`, `/reload-plugins`, `/todo` with its pinned todo row, natural-language `/loop`, and the "Model & Provider" section of `/status`. Others exist only when `experimental_enable_registry_skills` is on (`/skills`).

---

## 1. Master feature table

Classes:
- **UI**: UI-only. Works with any agent once wired.
- **CAP**: needs a backend capability (named generically).
- **CLOUD**: only makes sense with Mistral's hosted services.
- **DEV**: dev or debug only.

Where a feature mixes classes, the main class comes first.

### 1.1 Layout, chrome and status bar

| # | Feature | Vibe files (~lines) | Copied? | Backend used in Vibe | Class |
|---|---|---|---|---|---|
| 1 | Main layout: `ChatScroll#chat` (banner, `#messages`, `SubagentTranscripts`), `#loading-area` (NarratorStatus, spinner slot, InlineNotice, FeedbackBar), `#bottom-app-container` (ChatInputContainer), TodoStatusRow, `#bottom-bar` (path, PID, spacer, context gauge). | app.py `_compose_main_ui` L1103–1197; app.tcss 2,429 | ChatScroll yes; compose no; tcss yes (recolored copy) | runtime.context_window, runtime.stats, cwd | UI |
| 2 | Banner: brand, version, `model[thinking]`, models count, MCP servers enabled/total, connectors x/y, skills count, hooks count, plan description, "model pending" spinner, animated cat mascot (variant chosen by model name), `freeze_animation()` on first submit. | widgets/banner/banner.py 227, petit_chat.py 692, braille_renderer.py 58 | yes | config.active_model, config.models, runtime.mcp, runtime.connectors, runtime.custom_skills_count, runtime.hooks_count, account plan (`plan_title`), `awaiting_experiment_model` | CAP (model and integration counts); plan and pending-model parts are CLOUD |
| 3 | Bottom bar: cwd with `~` shortening, `[PID n]` label, quit-confirm prompt shown here. | widgets/path_display.py 31; cli/_process_title.py 15 | path yes; _process_title no | cwd | UI |
| 4 | Context-usage gauge: `12.3k/128k tokens (10%)`, per-subagent when viewing a subagent, reset after /clear and /compact. | widgets/context_progress.py 34; count_format.py 25; app.py `_refresh_context_progress` L5961 | yes (app wiring no) | `StatsUpdated` event (`context_tokens`, `context_window`) | CAP: token usage and context window |
| 5 | Loading spinner: status text, elapsed timer, hint "(Ns Esc to interrupt)" or "…Ctrl+C to cancel last queued message", queue count, typing-debounce hint, action-required state (paused timer), random easter-egg statuses (seasonal lists), gradient colors. | widgets/loading.py 327, spinner.py 195, spinner_text.py 49 | yes | none | UI |
| 6 | Inline notice next to the spinner (for example "Copied to clipboard" and queue-mode help). | widgets/inline_notice.py 41 | yes | none | UI |
| 7 | Toast notifications (`App.notify`) for `ServerWarning`, `ServerError`, config issues and similar. | app.py `_handle_immediate_turn_event` L3241 | no | events | UI |
| 8 | Terminal tab title: shows the session title; `>> ` while running and `? ` while waiting (when `experimental_enable_tab_status`); appends `- Action Required` or `- Task Complete` when blurred; throttled bell when blurred (when `enable_notifications`). | notifications/adapters/textual_notification_adapter.py 149, ports/notification_port.py 18 | no | session title (`SessionTitleUpdatedNoticeDetail`, `sessions.rename`) | UI (needs "session title") |
| 9 | Todo status row pinned above the input (one-line summary, click to open overlay), `TodoOverlayScreen` modal (Esc or q closes; items grouped in progress / pending / done / cancelled), `/todo`. | widgets/todo_status.py 30; screens/todo_overlay.py 58 + .tcss 36; todo_tracker.py 159 | tracker yes; row and overlay no | todo tool effects (`TodoEffectItem`), experimental harness only | CAP: todo-list tool events |

### 1.2 Startup flow

| # | Feature | Vibe files | Copied? | Backend | Class |
|---|---|---|---|---|---|
| 10 | Mount-first cold start: UI renders before the session exists, input is gated on `_session_ready`, bootstrap failure shows "Failed to start session: …" and disables input. | app.py `on_mount`, `_bootstrap_session`, `_show_bootstrap_error` L1223–1297; `run_textual_ui` L6653 | no | `AppServerHost.read_config`, `open_session`/`start_session` | CAP: async session start |
| 11 | "Initializing" spinner until background init is ready; fatal init shows "Background initialization failed" and "Press any key to exit…". | app.py `_watch_init_completion` L1623, `on_key` L1817 | no | `runtime.wait_until_ready` (`session/ready/wait`) | CAP: readiness signal |
| 12 | Workspace trust dialog, a separate Textual app run before the TUI. | textual_ui/startup.py 146; setup/trusted_folders/trust_folder_dialog.py 384 + .tcss 161 | no | `workspace/trust/status`, `workspace/trust/decision` | CAP: folder trust |
| 13 | Initial prompt from the CLI (`-p`-style positional) is auto-submitted after startup. | app.py `_process_startup_prompt` L1593 | no | turn submit | UI |
| 14 | `--resume` (picker), `--resume <id>`, `--continue`: start a fresh session, then rebind in place ("Resuming session…", "No previous sessions found."). | app.py `_auto_resume_on_startup` L4770; startup.py | no | `sessions.resolve_continue_session` (`session/continue`), `session/resume` | CAP: list, resume and last-session lookup |
| 15 | Greeting "Hello <first name>", at most once per 24 h (stored in the cache file), hidden when `show_greeting` is off. | app.py L6018–6066, L6350; messages.py `GreetingMessage` | widget yes | `identity.current.first_name` (Mistral identity); `FileSystemCacheStore` (utils/cache_store.py, needs tomllib/tomli_w) | CLOUD (adaptable to a local username) |
| 16 | What's New message shown once per version, plus a plan-offer CTA and a VS Code promo suffix. | update_notifier/whats_new.py 56, update.py 246; `vibe/whats_new.md`; messages.py `WhatsNewMessage` | widget yes, rest no | local file and cache; `account.current` for the plan CTA | UI (needs own content); plan CTA is CLOUD |
| 17 | VS Code extension promo in VS Code, Insiders and Cursor terminals (with a shown counter). | vscode_extension_promo/ (99 lines); terminal_detect.py 75; messages.py `VscodeExtensionPromoMessage` | widget yes | none | CLOUD (Mistral extension) |
| 18 | Background update check (PyPI, GitHub, `uv tool`) for package "mistral-vibe". | update_notifier/* (~1,000 lines total) | no | network to PyPI and GitHub | UI after retargeting to OVAT's package (otherwise CLOUD) |
| 19 | Config issues and validation-warning toasts (deduplicated). | app.py L1600–1621 | no | `runtime.issues`, `config.validation_warnings` | CAP: config validation |
| 20 | MCP discovery-failure toasts and the "MCP server X needs OAuth… run `/mcp login X`" notice. | app.py L1772–1800 | no | `runtime.mcp.discovery_errors`, `runtime.mcp.needs_auth` | CAP: MCP manager |
| 21 | "New version available for skill(s)… Run /skills" toast. | app.py L1754 | no | `skills.updates` (registry) | CLOUD |
| 22 | "You are using our new unified harness…" warning. | app.py L1694 | no | `runtime.experimental_harness` | DEV (Vibe-internal; drop) |
| 23 | Custom-tools deprecation message. | messages.py `CustomToolsDeprecationMessage`; app.py L6374 | widget yes | `runtime.tools[].is_custom` | CAP (Vibe-internal; probably drop) |
| 24 | Dangerous-directory warning (home, Documents, Desktop and similar). | app.py L6248; utils/paths.py 117 | no | local | UI |
| 25 | "Untrusted local config folders are being ignored" warning, once per folder. | app.py L6256 | no | `workspace/trust/untrustedConfig`; cache store | CAP: folder trust and config layering |
| 26 | Startup timing telemetry (`vibe.startup`). | app.py L1721 | no | `telemetry/record` | DEV |

### 1.3 Input composer

| # | Feature | Vibe files | Copied? | Backend | Class |
|---|---|---|---|---|---|
| 27 | Multi-line `ChatTextArea`: Enter submits, Shift+Enter or Ctrl+J newline, Alt+←/→ word moves, Shift+Backspace/Delete, cursor-blink preference, click-chain word select inside input. | widgets/chat_input/text_area.py 859, body.py 685, container.py 375 | yes | config.cursor_blink | UI |
| 28 | Input modes chosen by the first character on an empty line: `>` prompt, `/` command, `!` bash, `&` teleport. Shows the prompt glyph; Backspace resets the mode. | cli/input_modes.py 9; chat_input/input_kinds.py 59 | yes | none (`&` needs teleport) | UI |
| 29 | Persistent input history (Up/Down, history file). | cli/history_manager.py 151 | yes | none | UI |
| 30 | Slash-command completion popup (fuzzy), including user-invocable skills. | autocompletion/completers.py 569, fuzzy.py 214, slash_command.py 107; chat_input/completion_manager.py 74, completion_popup.py 90 | yes | `runtime.skills` | UI (skills part is CAP) |
| 31 | `@path` file completion backed by a background file index and watcher (gitignore-aware rules). | autocompletion/file_indexer/* (787), path_completion.py 205 | yes | config.file_watcher_for_autocomplete | UI |
| 32 | Inline skill completion. | autocompletion/inline_skill_completion.py 143 | yes | skills list | CAP: list skills |
| 33 | Paste handling: pasted image paths become `@path`; bare absolute image paths are rewritten. | chat_input/paste_path.py 160 | yes | none | UI |
| 34 | Clipboard image paste (Ctrl+V binding and `/paste-image`), **macOS only** (osascript and `sips`). Saves PNG to `$TMP/vibe-pasted-images` and inserts `@path`. | chat_input/paste_image.py 205; cli/constants.py 5 | paste_image no | `utils/images.MAX_IMAGE_BYTES` | UI (agent must accept images: CAP "image input") |
| 35 | External editor (Ctrl+G in input, `$EDITOR`). | textual_ui/external_editor.py 38 | yes | none | UI |
| 36 | Agent/mode indicator in the input border (agent name, safety color, "| auto-approve" suffix). Shift+Tab cycles agents with coalesced switching and a spinner after 0.5 s. Built-in agents: ask, plan, accept-edits, smart-approve, auto-approve, explore, lean (`vibe/core/agents/models.py`). | app.py L5928–6123; container.py `set_safety`, `set_agent_name` | container yes; app no | `agents.active`, `agents.next`, `agents.switch` (`session/agent/update`), `runtime.bypass_tool_permissions` | CAP: list and switch agent modes with a safety level |
| 37 | Typing-pause grace before an approval or question steals focus (`VIBE_TYPING_GRACE_PERIOD_MS`, default 1000 ms). | app.py `_wait_for_typing_pause` L3033 | no | none | UI |
| 38 | VS Code terminal space-key fix and `VscodeCompatInput`. | widgets/vscode_compat.py 26 | yes | none | UI |
| 39 | Prompt preparation: @-mention expansion, image attachments, auto-title, mention stats; errors shown inline. | app.py `_prepare_prompt_or_abort` L2214 | no | `workspace/prompt/prepare` → `PreparedPrompt` | CAP: prompt preprocessing (mentions and images) |
| 40 | `!command` runs a shell command directly, with streamed output in `BashOutputMessage`. Empty `!` gives an error; Esc cancels. | app.py `_handle_bash_command` L2875; messages.py `BashOutputMessage` | widget yes | `shell.run` (`session/shellCommand`, streamed events) | CAP: user shell exec with streaming |
| 41 | `/skillname args` invokes a skill as a prompt. | app.py `_resolve_skill` L2839 | no | `runtime.get_skill` | CAP: skills |

### 1.4 Transcript widgets

| # | Feature | Vibe files | Copied? | Backend | Class |
|---|---|---|---|---|---|
| 42 | `UserMessage`: image attachments row, pending (queued) style, severity (error), rewind highlight, `history_entry_id`, fired-loop tag. | widgets/messages.py 849 | yes | `PublicMessageEntry(role=user)` | UI |
| 43 | `AssistantMessage` streaming Markdown (append patches). | messages.py; handlers/event_handler.py 631 | yes | `HistoryEntryAdded/Updated` with `append` patches on `/content/0/text` | CAP: token streaming |
| 44 | `ReasoningMessage` (thinking): collapsible with a spinner; visibility follows `show_thinking_nodes`. | messages.py; app.py `_apply_thinking_visibility` L2402 | yes | `PublicReasoningEntry` | CAP: reasoning stream |
| 45 | Tool calls: `ToolGroup` with header, `ToolCallMessage`, `ToolResultMessage`, and per-tool renderers for bash/process, write file, edit (diff), read, grep, todo, ask-user-question, web search, web fetch, scratchpad, plus a generic fallback. | widgets/tools.py 866, tool_widgets.py 786, utils/tool_presentation.py 90 | yes | `PublicEffectEntry` with typed inputs and outputs | CAP: structured tool-call events |
| 46 | Diff rendering (ANSI or truecolor, restyled when the theme changes). | widgets/diff_rendering.py 401; app.py `_restyle_diff_widgets` L2717 | yes | none | UI |
| 47 | Collapsible sections; Ctrl+O toggles all tool output. | widgets/collapsible.py 314; app.py `action_toggle_tool` L5921 | yes | none | UI |
| 48 | `SlashCommandMessage` echo and `UserCommandMessage` (Markdown system replies). | messages.py | yes | none | UI |
| 49 | `ErrorMessage` (with a retry offer), `WarningMessage`, `InterruptMessage` ("Interrupted | What should Vibe do instead?"). | messages.py | yes | none | UI |
| 50 | Hook run display (`HookRunContainer`, `HookSystemMessageLine`) for hook_run_started/completed and hook_started/completed. | messages.py; event_handler.py L453–520 | yes | `HookNoticeDetail` notices | CAP: hooks |
| 51 | Plan mode: `PlanFileMessage`, plan review start/end (custom input border), Ctrl+G opens the plan in the editor, clear-context-on-plan-accept ("Context cleared. Implementing the approved plan..."). | messages.py `PlanFileMessage`; event_handler.py L590; app.py `_on_context_cleared` L4953, `action_open_plan_in_editor` L6571 | widgets yes; app no | `PlanReviewStarted/Ended`, `ContextCleared` notices | CAP: plan mode |
| 52 | Model-change checkpoint message. | widgets/model_change.py 33 | yes | `PublicCheckpointEntry(kind="model_change")` | CAP |
| 53 | `CompactMessage` (spinner, then done or error). | widgets/compact.py 26 | yes | compaction checkpoint | CAP: compaction |
| 54 | `ReloadConfigMessage`, `BranchCreatedMessage`, `RewindForkMessage`, `TeleportMessage` (all StatusMessage subclasses). | reload_message.py 33, branch_created_message.py 21, rewind_fork_message.py 21, teleport_message.py 46; status_message.py 104; utils/display.py 22 | status_message yes; others no | see #84, #86, #88, #112 | UI shells |
| 55 | Clickable links with URL normalization in `App.open_url`. | widgets/links.py 86; app.py L717 | yes (override no) | none | UI |
| 56 | Session-title and agent-changed notices update the tab title and indicators. | event_handler.py L396–420 | yes | notices | CAP |

### 1.5 Turn lifecycle

| # | Feature | Vibe files | Copied? | Backend | Class |
|---|---|---|---|---|---|
| 57 | Submit: optimistic "turn in flight" (instant spinner), enqueue, server promotes, `TurnStarted`/`TurnCompleted`; unsolicited turns (loops) also render. | app.py L2913–2938, `_listen_app_server_events` L3288, `_begin/_complete_unsolicited_turn` L3309–3354 | no | `session/turn/enqueue`, event stream | CAP: async turns with an event stream |
| 58 | Interrupt (Esc): status changes to "Interrupting" immediately; waits for a pending turn to surface; "settle" window so a racing submit sees the post-interrupt state; toast after 2 s "Interrupt is taking unusually long. Press Ctrl+C twice to force quit."; then `InterruptMessage`. | app.py `_interrupt_turn` L3830, `_interrupt_server_turn` L3912, settle helpers L5656–5700 | no | `turn/interrupt` | CAP: interrupt |
| 59 | Turn error messages by code: rate limit (with an "upgrade to Pro" variant), context too long (suggests /rewind and /compact), refusal (category and explanation), retryable codes get a "/retry" hint; Sentry for non-benign codes. | app.py L3554–3610 | no | `TurnErrorCode`, `account.rate_limit_action` | CAP: typed error codes; Pro CTA is CLOUD |
| 60 | Silent auto-retry on incomplete stream (2 attempts), reusing the partial assistant message; `/retry [instructions]`. | app.py `_auto_retry_incomplete_stream` L3477, `_retry` L3500; event_handler `offer_retry/begin_retry`; utils/retry_prompt.py 17 | partial | injected turn (`act(..., injected=True)`) | CAP: continue/retry a turn |
| 61 | Completion notification on turn end (bell and title). | app.py `_finalize_turn_ui` L3522 | no | none | UI |
| 62 | History windowing: on resume, mount only the last 20 entries; "Load more messages (N)" button loads 10 per click and pages older history from the server. | windowing/history.py 294 (copied), state.py 117, history_windowing.py 59, widgets/load_more.py 43; app.py L5857–5919 | history.py yes; others no (OVAT's `windowing/__init__.py` is empty) | `sessions.history_before_cursor`, `load_before` (`session/history/list`) | CAP: paged history |
| 63 | DOM pruning: when virtual height exceeds 1,500 lines, keep about 1,000 (`prune_oldest_children`). | app.py L486–539, `_try_prune` L6474 | no | none | UI |
| 64 | Scroll behavior: smooth wheel glide, auto re-anchor at bottom, anchoring suppressed while selecting, Shift+↑/↓ scroll 5 lines, per-transcript scroll memory. | app.py `ChatScroll` L411–483, L6236 | ChatScroll yes | none | UI |
| 65 | GC tuning: `gc.freeze()` after mount, `PAUSE_GC_ON_SCROLL`. | app.py L679, L1359 | no | none | UI |

### 1.6 Message queue and steering

| # | Feature | Vibe files | Copied? | Backend | Class |
|---|---|---|---|---|---|
| 66 | Queue while busy: prompts typed during a turn merge into **one** queued turn (`QueueController`); `QueueHeaderMessage`; queued count in the spinner; quit warning "N queued messages will be discarded". | textual_ui/message_queue.py 953; messages.py `QueueHeaderMessage` | queue no; header yes | `session/turn/enqueue`, `…/queue/replace`, `…/queue/remove`, `…/queue/resume`, `…/queue/read`, `TurnQueueUpdated` | CAP: turn queue (could be emulated client-side) |
| 67 | Queue selection and edit mode: Up from input enters it; Up/Down select, Enter edits, Backspace/Delete removes, Esc exits; inline help notice; edits copy-on-write if the item was already promoted. | chat_input/body.py L245–480; app.py L1881–1983 | body yes; app no | queue replace/remove | CAP (as #66) |
| 68 | Steering: Ctrl+Enter (or Super+Enter), or Enter on empty input, injects queued prompts into the running turn. | app.py `_steer_queued_now` L2242; message_queue.py `steer_pending` | no | `session/context/inject` (`require_active_turn`), `session/turn/queue/steer` | CAP: mid-turn user message injection |
| 69 | Queue pauses after an interrupt; the next submit resumes it. | app.py `_handle_paused_submit` L2042 | no | `queue/resume`, `paused` flag | CAP |
| 70 | Ctrl+C removes the last queued message before it interrupts or quits. | app.py `action_interrupt_or_quit` L6143 | no | queue remove | CAP |
| 71 | Side-channel commands allowed while busy: help, copy, paste-image, log, debug, exit, status, whoami, rename, data-retention (one at a time). Other slash, bash and teleport input is rejected with a warning while busy. | message_queue.py `SideChannelController` L895–953; app.py L2000, L2050 | no | none | UI |

### 1.7 Approvals and questions

| # | Feature | Vibe files | Copied? | Backend | Class |
|---|---|---|---|---|---|
| 72 | `ApprovalApp`: tool approval with Yes, Always-this-tool-for-session, Always-permanently, No (keys 1–4, y, n); path-scope choices; required-permission list; reason; diff preview for edits; grace period against accidental keypresses. | widgets/approval_app.py 325; tool_widgets.py approval widgets | yes (OVAT already uses it) | `CallbackRequested` with `ApprovalCallbackDetail`; `callback/result` with `ApprovalDecision(APPROVE, APPROVE_FOR_SESSION, APPROVE_PERMANENTLY, DENY, path_scope)` | CAP: approval callbacks with session and persistent grants |
| 73 | Callback queue: several pending approvals or questions shown one after another; spinner enters the "action required" state; terminal "Action Required" notify. | app.py `_show_callback` L3076, `_respond_to_active_callback` L3139 | no | callbacks | UI |
| 74 | `QuestionApp` for the ask-user-question tool: several questions as tabs (←/→), single or multi-select, "Other" free text, number-key quick select, Submit row, grace period, Esc cancels. | widgets/question_app.py 619 | no | `UserInputCallbackDetail` → `UserInputCallbackOutput(UserQuestionResult)` | CAP: ask-user tool |
| 75 | Local questions asked by the UI itself (teleport push confirmation) reuse `QuestionApp`. | app.py `_request_local_user_input` L3056 | no | none | UI |

### 1.8 Session management (slash commands)

| # | Feature | Vibe files | Copied? | Backend | Class |
|---|---|---|---|---|---|
| 76 | `/resume` (`/continue`) session picker: sessions for the cwd sorted by updated time; relative time, short id, `[harness]` tag, title or preview; **live preview** of the highlighted session's history in the transcript (debounced 0.1 s); Enter resumes; `d` deletes (with confirmation and feedback; the current session cannot be deleted); Esc cancels and restores the live transcript. | widgets/session_picker.py 500; app.py L4465–4670 | no | `session/list`, `session/read`/history get, `session/resume`, `session/delete` | CAP: list, preview, resume and delete saved sessions |
| 77 | In-place resume: reset presentation, rebuild transcript, sync queue, show "Resumed session `abcd1234`", post-init notices after a fast resume; a "session created by a newer version" error gets an upgrade hint. | app.py `_resume_local_session` L4672, `_reset_presentation_after_resume` L4700; textual_ui/_resume_errors.py 43 | no | `session/resume` | CAP |
| 78 | `/clear` (`/new`) `[prompt]`: new conversation; prints the previous session id and "To resume it later, run: `vibe --resume <id>`"; optional seed prompt. | app.py `_clear_history` L4908 | no | `session/history/clear`; session_log enabled/persisted | CAP: new session |
| 79 | `/compact [instructions]`: refused while busy or when history is empty; shows `CompactMessage`; clears the queue. | app.py L5000–5050 | no | `session/compact` | CAP: summarize/compact context |
| 80 | `/rename <title>` (also updates the tab title). | app.py L4411 | no | `session/rename` | CAP |
| 81 | `/branch`: fork into a new resumable session, leaving the current one unchanged. | app.py L4434 | no | `session/fork` (`attach=False`) | CAP: fork session |
| 82 | `/rewind` and Esc-Esc on empty input: rewind mode. Steps through user messages (the highlight scrolls; going up past the top triggers Load More). `RewindApp` is a two-step choice: first "Edit & restore files to this point" or "Edit without restoring files", then "Keep in current session" or "Fork to a new session". Afterwards the transcript rebuilds and the message text goes back into the input; file-restore errors become toasts. Keys: ↑↓/jk pick, ←/→ previous/next message, 1/2, Enter, Esc back, q quit. | widgets/rewind_app.py 261; app.py L5406–5632 | no | `session/rewind/read` (has file changes), `session/rewind` (`restore_files`, `inplace`) | CAP: rewind to message, file checkpoints, fork |
| 83 | `/log`: show the session log directory. | app.py L4968 | no | `session/log/read` | CAP: session log path |
| 84 | `/status`: Agent Statistics (steps, session prompt/completion/total tokens, cached tokens, last-turn tokens, cost in $) and, on the unified harness, Model & Provider (model, provider, API base). | app.py L4221–4263; textual_ui/provider_auth_status.py 65 | no | `runtime.stats`, `providerAuth/read` | CAP: usage, cost and provider info |
| 85 | `/help`: keyboard shortcuts and sorted command list. | commands.py `get_help_text` | yes | none | UI |
| 86 | `/copy`: copy the last agent message. | app.py L3951 | no | none | UI |
| 87 | `/exit` (also `exit`, `quit`, `:q`, `:quit` without a slash) and an exit summary (session id and token usage) after shutdown, with "Closing…". | app.py `_exit_app` L5057, `_run_app_with_cleanup` L6620; cli/session_exit.py 28 | no | `exit_summary()` | UI (summary is CAP: usage) |

### 1.9 Settings, pickers and configuration

| # | Feature | Vibe files | Copied? | Backend | Class |
|---|---|---|---|---|---|
| 88 | `/config`: full-screen searchable settings browser (`ConfigScreen` ModalScreen). Type to filter; Popular and Advanced sections; a per-row origin layer and admin-enforced legend; Enter opens an edit modal with a "WHERE IT'S SET" inspector and a save-target toggle (Tab: session override vs TOML); Ctrl+S saves text fields; Ctrl+R resets to default; dynamic choices (themes, models); config reloads on close if dirty. | screens/config/config_screen.py 326, edit.py 249, _common.py 328, .tcss 169 | no | `config/fields/read`, `config/write` (JSON-patch ops), `config/reload` | CAP: layered config schema and write |
| 89 | `/model`: model picker with "Default (currently X)" plus configured models; blocked when an admin enforces `active_model`; persisting may stay pending until the turn ends. | widgets/model_picker.py 131; app.py L2441–2477, L5183 | no | `config.models`, `config/model/write`, `config/reload` | CAP: list and switch model |
| 90 | `/thinking`: thinking-level picker (levels per active model). | widgets/thinking_picker.py 89 | no | `config.set_thinking` | CAP: reasoning-effort levels |
| 91 | `/theme`: theme picker with live preview (debounced 0.1 s), "auto" theme, revert on Esc, persisted. | widgets/theme_picker.py 121; cli/theme.py 19; cli/_theme_detection.py 245 (OSC 11 and OS appearance) | no | `config.update({"theme"})` | UI (persist to OVAT config) |
| 92 | `/reload`: reload config, agent instructions and skills; warns "N images … will be omitted" for models without vision. | app.py `_reload_config` L4838 | no | `config/reload` (`reload_runtime=True`) | CAP: hot reload |
| 93 | `/proxy-setup`: form for proxy and SSL-cert keys (one input per key with description); "Restart the CLI for changes to take effect." | widgets/proxy_setup_app.py 114 | no | `config/proxy/read`, `config/proxy/write` | CAP: proxy settings (local) |
| 94 | `/log-level`: log-level picker with session-override and config.toml badges (←/→ switch badge, Enter toggle), aware of the `LOG_LEVEL` and `DEBUG_MODE` environment variables. | widgets/log_level_picker.py 225; observability/logging.py (copied) | picker no | `config.write` remove/update `log_level` | DEV |
| 95 | `/voice`: voice settings (voice mode On/Off, narrator On/Off); applies to the voice manager; "Press Ctrl+R to start recording". | widgets/voice_app.py 180; app.py L2398–2440, L2310 | no | `config.update` | See #106 and #107 (CLOUD) |
| 96 | Config-change subscription: theme, cursor blink, log level, banner and subagent list follow config live. | app.py `_on_config_changed` L1299 | no | `config.subscribe` | CAP: config change events |

### 1.10 Integrations

| # | Feature | Vibe files | Copied? | Backend | Class |
|---|---|---|---|---|---|
| 97 | `/mcp` (`/connectors`) browser: two panes (sources, then tools of the selected source), search (← focuses search), d/e disable/enable a server or single tool (optimistic toggle), r refresh (waits for init, refreshes connectors then servers), Enter on a source that needs auth opens the OAuth or connector-auth app, Backspace goes back, Esc closes. `/mcp <name>` opens a given source. | widgets/mcp_app.py 675; app.py L4126–4167, L2724–2785 | no | `mcp_catalog/read`, `…/refresh`, `…/toggle`, `connectors/refresh` | CAP: list MCP servers and tools, enable/disable |
| 98 | `/mcp add <url> [--name] [--scope …] [--transport http\|streamable-http] [--no-login] [--allow-insecure-http]` (OAuth-only shortcut; `--help`). | textual_ui/mcp_commands.py 135; utils/mcp.py (copied) | mcp_commands no | `mcp_catalog/add` | CAP: add MCP server |
| 99 | `/mcp status`, `/mcp login <alias>` (prints the URL and opens the browser; connectors open the connector app), `/mcp logout <alias>`. | app.py L3981–4081 | no | `mcp_catalog/login` (streams `mcp/authUrl`), `…/logout`, `…/read` statuses | CAP: MCP OAuth |
| 100 | `MCPOAuthApp`: open, copy or show the auth URL; r retries; waits for login to finish. | widgets/mcp_oauth_app.py 217 | no | `mcp.login` stream | CAP: MCP OAuth |
| 101 | `ConnectorAuthApp`: authenticate a **workspace connector** (open, copy or show URL; r refreshes). | widgets/connector_auth_app.py 217 | no | `connectors/auth/read`, `connectors/refresh` | CLOUD (Mistral workspace connectors) |
| 102 | `/plugins` browser (search with `/`, r reload, Backspace back) and `/reload-plugins` (diff report of content digests). Experimental harness only. | widgets/plugins_app.py 314; app.py L4169–4219 | no | `plugin_catalog/read`, `plugin/reload` | CAP: plugin catalog |
| 103 | `/skills` browser (registry skills flag only): Installed and Catalog tabs (←/→), search (`/`), preview pane, `v` versions, import with scope (global/project), pin version, latest or alias, `x` remove, `t` enable/disable; "Sign in to Mistral to browse shared skills". | widgets/skills_browser.py 964; app.py L4373–4401 | no | `skills/installed`, `/catalog`, `/detail`, `/versions`, `/import`, `/setVersion`, `/setLatest`, `/setAlias`, `/remove`, `/setEnabled`, `/updates` | CLOUD (Mistral skills registry); enable/disable of local skills is CAP |
| 104 | `/loop`: scheduled recurring prompts. `/loop <interval> <prompt>`, `/loop list`, `/loop cancel <id\|all>`; a Markdown table of loops; fired loops render as user prompts tagged `FiredLoop`. On the unified harness the arguments are forwarded to the model as natural language. | textual_ui/scheduled_loop_runner.py 94; widgets/fired_loop.py 57 | runner no; FiredLoop yes | `loops/list`, `loops/create`, `loops/delete`, `loops/clear`; `ScheduledLoopFiredNoticeDetail` | CAP: scheduler for recurring prompts |
| 105 | Subagents: a status list under the input (Down from the last input line focuses it; j/k; per-subagent status). Selecting one shows its **read-only transcript** (LRU of 5 cached trees, parent instruction shown first), with its own spinner and context gauge. Typing or Ctrl+C there shows "You can't interact with a subagent directly…"; "ready for a new instruction" appears on the idle transition; Esc returns to the main chat. Config `show_subagent_status_list`. | chat_input/subagent_list.py 318 (copied); widgets/subagent_transcripts.py 249; app.py L1370–1558, L5796–5818 | list yes; transcripts no | `child_sessions`, `ChildSessionUpdated`, `sessions.get_session_history`, `list_history` | CAP: child sessions with history, status and context usage |

### 1.11 Voice and narrator

| # | Feature | Vibe files | Copied? | Backend | Class |
|---|---|---|---|---|---|
| 106 | Voice input: Ctrl+R starts recording when voice mode is on, any key stops, Esc or Ctrl+C cancels; recording indicator with a peak meter; live streaming transcription into the input; errors and notices inline. | voice_manager/voice_manager.py 337 (port copied), audio_recorder/ 377, transcribe/ 187 (`MistralTranscribeClient` via `mistralai.extra.realtime`), lazy_audio_managers.py 264, audio_request_metadata.py 25; recording/recording_indicator.py 80 (copied); chat_input/body.py recording UI (copied) | ports and indicator yes; implementation no | Mistral realtime transcription API (API key), `miniaudio` | CLOUD (could be re-targeted to a local STT) |
| 107 | Narrator (experimental): after each turn, summarizes it (`narration/summarize`, an LLM call on the server) and speaks the summary with Mistral TTS; `NarratorStatus` animation "summarizing" or "speaking | Esc/Ctrl+C to stop". | narrator_manager/ 378, tts/ 126, audio_player/ 254, turn_summary/ 212; widgets/narrator_status.py 90 | no | `narration/summarize`, Mistral TTS API, `miniaudio` | CLOUD |

### 1.12 Mistral-cloud-only product features

| # | Feature | Vibe files | Copied? | Backend | Class |
|---|---|---|---|---|---|
| 108 | `/teleport`, `&prompt` and `--teleport`: move the session to "Vibe Code Web". Steps: summarize context, check git, push approval via QuestionApp ("You have N unpushed commits. Push to continue?"), start the workflow, then show a link. | app.py L3612–3828; widgets/teleport_message.py 46; messages.py `TeleportUserMessage` | partial | `vibeCode/teleport/start`, `/event`, `/push/respond`, `/cancel` | CLOUD |
| 109 | `/remote-project`: Vibe Code project picker (search with `/`, match kinds, load more, unlink, create project with name and default branch, stale-link recovery). | widgets/vibe_code_project/picker.py 614, create.py 93, state.py 39, __init__ 19 | no | `vibeCode/projects/open`, `/loadMore`, `/create`, `/select`, `/unlink`, `/cancel`, `/recover` | CLOUD |
| 110 | `/whoami`: name, email, workspace, organization and plan of the signed-in Mistral account. | app.py L4265; plan_offer/presentation.py 18 | no | `identity/read`, `account/read` | CLOUD |
| 111 | `/data-retention`: "Your Data Helps Improve Mistral AI" with a chat.mistral.ai link. | utils/data_retention.py 9 | no | none | CLOUD |
| 112 | `/leanstall` and `/unleanstall`: install or uninstall the "lean" (leanstral) agent. | app.py L4863–4891 | no | `agents/install`, `agents/uninstall` | CLOUD (the generic "install agent profile" capability is CAP) |
| 113 | Feedback bar "How's your session with Vibe? 1: good 2: fine 3: bad 0: snooze" (keys captured by ChatTextArea; "Thank you…" and "Snoozed for …" replies); shown based on a server-side cadence. | widgets/feedback_bar.py 85; text_area.py FEEDBACK keys; vibe/feedback.py 3 | text_area yes; bar no | `feedback/shouldShow`, `feedback/record`, `telemetry/record` (`vibe.user_rating_feedback`) | CLOUD |
| 114 | Telemetry events: `vibe.slash_command_used`, `at_mention_inserted`, `user_copied_text`, `user_cancelled_action`, `voice_mode_toggled`, `session_branched`, `startup`. | app.py (many) | no | `telemetry/record` | CLOUD / DEV |
| 115 | Sentry crash reporting (`capture_sentry_exception` in `_handle_exception`, worker errors, non-benign turn errors). | observability/sentry.py 233 | no | `sentry_sdk` | DEV / CLOUD |
| 116 | Onboarding app (separate Textual app before first run): welcome, theme selection, auth method, API key, browser sign-in, sign-in target, custom domain. | setup/onboarding/ ~2,600 lines + setup/auth/ ~1,550 | no | Mistral console sign-in, API key keyring | CLOUD (theme selection and API-key screens are adaptable) |
| 117 | Update prompt dialog (`--check-upgrade`, startup update prompt). | setup/update_prompt/ ~390 | no | update gateways | UI after retargeting |
| 118 | Admin-enforced config (`ADMIN_LAYER` origin; blocks `/model` when enforced). | screens/config/_common.py; app.py `_is_active_model_enforced` L5176; app_server/_admin_config.py | no | `config/fields/read` origins | CLOUD (enterprise admin config) |
| 119 | "Model pending" (`awaiting_experiment_model`) spinner in the banner. | banner.py; app.py `_model_pending` L967 | partial | experiments service | CLOUD |

### 1.13 Clipboard, selection and terminal plumbing

| # | Feature | Vibe files | Copied? | Backend | Class |
|---|---|---|---|---|---|
| 120 | Copy selection with Ctrl+Y or Ctrl+Shift+C; autocopy on mouse-up (`autocopy_to_clipboard`); OSC 52 over SSH or when pyperclip fails; inline notice "Copied | <native hint>". | cli/clipboard.py 177; app.py L6522–6556 | no | none | UI |
| 121 | Multi-click selection: double-click selects a word, triple-click a paragraph, dragging extends by that unit. This is the default screen `WordSelectScreen`. | textual_ui/word_selection.py 465 | no | none | UI |
| 122 | Terminal input filter: drops malformed SGR mouse reports (VS Code `NaN`), drops OSC 11 colour replies, re-enables SGR mouse mode after a legacy X10 report (tmux reattach). Installed by patching `XTermParser` in `get_driver_class`. | textual_ui/terminal_input_filter.py 163 | no | none | UI |
| 123 | Auto theme: OSC 11 background query and OS appearance (macOS, Windows, GNOME and others); `resolve_auto_theme` before the app runs. | cli/_theme_detection.py 245; cli/theme.py 19 | no | none | UI |
| 124 | Ctrl+Z suspends with the message "Mistral Vibe has been suspended. Run fg…"; forced repaint on SIGCONT. | app.py L6578–6590 | no | none | UI |
| 125 | Quit confirmation: Ctrl+C or Ctrl+D twice within 1 s; the prompt appears in the path bar along with the queued-message count; `ask_confirmation_on_exit`. Ctrl+D deletes right when the input is non-empty. Ctrl+C priority: clear input, then confirmed quit, then Esc-ladder steps, then pop queued item, then interrupt job, then ask to quit. | textual_ui/quit_manager.py 65; app.py L6143–6213 | no | none | UI |
| 126 | Escape priority ladder: subagent view → main; cancel voice recording; dismiss completion or clear `/…` input; close or cancel the current bottom app (per-app handlers, honoring approval and question grace periods); Esc-Esc within 0.2 s clears input or enters rewind; cancel narrator; interrupt bash or agent. Disabled on modal screens and in queue mode via `check_action`. | app.py L5641–5855 | no | none | UI |
| 127 | Shutdown hygiene: SIGTERM forces quit; `stderr_guard`; "Closing…" with cleanup (voice, narrator, app-server close); events listener stopped before DOM teardown. | app.py L3261–3302, L6215, L6620; cli/stderr_guard.py 101 | no | none | UI |
| 128 | Process name "Vibe CLI" and the `[PID n]` tag. | cli/_process_title.py | no | none | UI (rebrand) |
| 129 | VS Code-family terminal detection. | cli/terminal_detect.py 75 | no | none | UI |
| 130 | `/debug` and Ctrl+\\: debug console, a docked log viewer (virtualized ScrollView with wrap, colored levels, loads older pages on scroll-up, polls every 0.5 s). | widgets/debug_console.py 289 | no | `diagnostics/logs/read` (`runtime.read_logs`) | DEV |
| 131 | Review resource (`review/state`, `/baseline`, `/turnDiff`, `/hunks`, `/approve`, `/revert`). | app_server/_session_resources.py `ReviewResource`; app_server/review.py (copied) | models yes | review RPCs | **Not used by the TUI** (no `resources.review` call anywhere in textual_ui). Listed for completeness; not a TUI feature. |

**Total features: 130** (#1–#130; #131 is not a TUI feature). Counted by main class:

| Class | Count | Feature #s |
|---|---|---|
| **UI** (UI-only) | 49 | 1, 3, 5–8, 13, 16, 18, 24, 27–31, 33–35, 37, 38, 42, 46–49, 54, 55, 61, 63–65, 71, 73, 75, 85–87, 91, 117, 120–129 |
| **CAP** (backend capability needed) | 58 | 2, 4, 9–12, 14, 19, 20, 23, 25, 32, 36, 39–41, 43–45, 50–53, 56–60, 62, 66–70, 72, 74, 76–84, 88–90, 92, 93, 96–100, 102, 104, 105 |
| **CLOUD** (Mistral hosted) | 18 | 15, 17, 21, 95, 101, 103, 106–114, 116, 118, 119 |
| **DEV** | 5 | 22, 26, 94, 115, 130 |

---

## 2. Slash commands (registry: `vibe/cli/commands.py`, 338 lines, copied)

Columns: Command | Aliases | Handler in app.py | Availability | Side-channel (allowed while busy) | Feature #.

| Command | Aliases | Handler | Availability | Side-ch. | # |
|---|---|---|---|---|---|
| help | /help | `_show_help` | always | yes | 85 |
| config | /config | `_show_config` | always | | 88 |
| model | /model | `_show_model` | always | | 89 |
| skills | /skills | `_show_skills` | `registry_skills_enabled` | | 103 |
| thinking | /thinking | `_show_thinking` | always | | 90 |
| reload | /reload | `_reload_config` | always | | 92 |
| clear | /clear, /new | `_clear_history` | always | | 78 |
| copy | /copy | `_copy_last_agent_message` | always | yes | 86 |
| paste-image | /paste-image | `_paste_clipboard_image_command` | macOS only | yes | 34 |
| log | /log | `_show_log_path` | always | yes | 83 |
| log-level | /log-level | `_log_level_command` | always | | 94 |
| debug | /debug | `action_toggle_debug_console` | always | yes | 130 |
| compact | /compact | `_compact_history` | always | | 79 |
| exit | /exit, exit, quit, :q, :quit | `_exit_app` | always | yes | 87 |
| status | /status | `_show_status` | always | yes | 84 |
| whoami | /whoami | `_show_whoami` | always | yes | 110 |
| teleport | /teleport | `_teleport_command` | always (also `&` prefix) | | 108 |
| remote-project | /remote-project | `_vibe_code_project_command` | always | | 109 |
| proxy-setup | /proxy-setup | `_show_proxy_setup` | always | | 93 |
| resume | /resume, /continue | `_show_session_picker` | always | | 76 |
| rename | /rename | `_rename_session` | always | yes | 80 |
| mcp | /mcp, /connectors | `_show_mcp` (+ add/status/login/logout) | always | | 97–100 |
| plugins | /plugins | `_show_plugins` | experimental harness | | 102 |
| reload-plugins | /reload-plugins | `_reload_plugins` | experimental harness | | 102 |
| todo | /todo | `_show_todos` | experimental harness | | 9 |
| voice | /voice | `_show_voice_settings` | always | | 95 |
| leanstall | /leanstall | `_install_lean` | always | | 112 |
| unleanstall | /unleanstall | `_uninstall_lean` | always | | 112 |
| rewind | /rewind | `_start_rewind_mode` | always | | 82 |
| branch | /branch | `_branch_session` | always | | 81 |
| retry | /retry | `_retry` | always | | 60 |
| loop | /loop | `_loop_command` (`forwards_to_model` on experimental harness) | always | | 104 |
| data-retention | /data-retention | `_show_data_retention` | always | yes | 111 |
| theme | /theme | `_show_theme` | always | | 91 |

That is 34 commands. Dispatch works as follows: `CommandRegistry.parse_command` matches aliases (non-slash aliases only when there are no arguments), `classify()` in `input_kinds.py` decides the input kind, and `_invoke_resolved_command` (app.py L2814) mounts a `SlashCommandMessage` and calls `getattr(self, command.handler)(cmd_args=…, command_message=…)`. User-invocable skills also appear as `/name` entries in completion (#30, #41). Help text: `CommandRegistry.get_help_text`.

---

## 3. Key bindings

### 3.1 App level (`VibeApp.BINDINGS`, app.py L681–698)

| Key | Action | Notes |
|---|---|---|
| Ctrl+C | `interrupt_or_quit` | Priority ladder (#125) |
| Ctrl+D | `delete_right_or_quit` (priority) | |
| Ctrl+Z | `suspend_with_message` (priority) | Not on Windows |
| Escape | `interrupt` (priority) | Ladder (#126); disabled on ModalScreen and in queue mode |
| Ctrl+O | `toggle_tool` | Collapse or expand all tool output |
| Ctrl+Y, Ctrl+Shift+C | `copy_selection` (priority) | |
| Shift+Tab | `cycle_mode` (priority) | Next agent |
| Shift+↑ / Shift+↓ | scroll chat ±5 lines (priority) | |
| Ctrl+G | `open_plan_in_editor` | The ChatTextArea Ctrl+G (external editor) wins when the input is focused |
| Ctrl+\\ | `toggle_debug_console` | |
| any key | exit, when a fatal init error is shown | `on_key` |
| mouse up | autocopy selection | `on_mouse_up` |

### 3.2 ChatTextArea (`widgets/chat_input/text_area.py` L60–90 and `_on_key` L584–720)

| Key | Action |
|---|---|
| Enter | submit (or accept completion) |
| Shift+Enter, Ctrl+J | newline |
| Ctrl+Enter, Super+Enter | steer queued prompts (only when input is empty) |
| ↑ / ↓ | history; ↑ enters queue selection when queue items exist; ↓ on the last wrapped line moves to the subagent list |
| Alt+← / Alt+→ | word left or right |
| Shift+Backspace, Shift+Delete | delete left or right |
| Ctrl+G | external editor |
| Ctrl+V | paste clipboard image (macOS only) |
| Ctrl+R | start voice recording (voice enabled); while recording any key stops and Ctrl+C cancels |
| `!` `/` `&` on empty input | switch input mode; Backspace resets it |
| 1, 2, 3, 0 | feedback rating or snooze while the feedback bar is shown |
| Queue mode: ↑↓, Enter, Backspace/Delete, Esc | select, edit, remove or exit queued item |
| Esc while editing a queued item | cancel the edit |
| Tab and others | completion manager (`CompletionResult.HANDLED`/`SUBMIT`) |

### 3.3 Bottom apps and screens

| Widget | Keys |
|---|---|
| ApprovalApp (copied) | ↑↓ (+ j/k via VimNavigationMixin), Enter, 1/y yes, 2 always for this tool in the session, 3 always permanently, 4/n no; Esc rejects (after grace) |
| QuestionApp | ↑↓, j/k, Enter select, Space toggle (multi), digits quick-select, ←/→ switch question, Esc cancel |
| RewindApp | ↑↓, j/k, ←/→ previous/next message, 1/2, Enter, q quit, Esc back |
| SessionPickerApp | ↑↓, j/k, Enter, d delete, Esc |
| ModelPicker, ThinkingPicker, ThemePicker | ↑↓, j/k, Enter, Esc (theme previews while moving) |
| LogLevelPicker | ↑↓, j/k, ←/h →/l switch badge, Enter toggle, Esc apply and close |
| VoiceApp | ↑↓, j/k, Space or Enter toggle, Esc close (saves) |
| ProxySetupApp | ↑↓ field focus, Enter save and exit, Esc cancel |
| MCPApp | Esc, Backspace back, d disable, e enable, r refresh, ← focus search (in list), Enter |
| MCPOAuthApp, ConnectorAuthApp | Esc or Backspace close, r retry/refresh, Enter on open/copy/show options |
| PluginsApp | Esc, Backspace, `/` filter, r reload |
| SkillsBrowserApp | Esc, Backspace back, v versions, x remove, t on/off, `/` search, ←/→ tabs |
| VibeCodeProjectPicker / Create | Esc, Enter, `/` search |
| ConfigScreen (modal) | type to filter, ↑↓, Enter edit, Ctrl+R reset, Esc close |
| Config edit modal | Tab switch save target, Esc cancel, Ctrl+S save (text) |
| TodoOverlayScreen | Esc or q |
| NavigableOptionList (copied) | j/k |
| SubagentList (copied) | NavigableOptionList keys, Enter select |
| Mouse | double or triple click word/paragraph selection (#121); click a tool header or reasoning block to collapse or expand; click the todo row |

---

## 4. Screens, modals, overlays, pickers and browsers

- **ModalScreens (pushed):** `ConfigScreen` (+ `_TargetedEditScreen` subclasses in `screens/config/edit.py`) and `TodoOverlayScreen`.
- **Default screen:** `WordSelectScreen` (`word_selection.py`).
- **Bottom apps:** ApprovalApp, QuestionApp, ModelPickerApp, ThinkingPickerApp, ThemePickerApp, LogLevelPickerApp, ProxySetupApp, VoiceApp, SessionPickerApp, RewindApp, MCPApp, MCPOAuthApp, ConnectorAuthApp, PluginsApp, SkillsBrowserApp, VibeCodeProjectPickerApp, VibeCodeProjectCreateApp.
- **Docked overlays:** DebugConsole (mounted on the app), FeedbackBar, InlineNotice, NarratorStatus, CompletionPopup (copied).
- **Separate pre-TUI apps:** TrustFolderApp (`setup/trusted_folders`), onboarding (`setup/onboarding`), update prompt (`setup/update_prompt`).

---

## 5. Python-version issues in files not yet copied (OVAT supports 3.10)

Method: `ast.parse` under `/usr/bin/python3.11` (to catch 3.12-only syntax), plus a regex scan for 3.11+ stdlib names, over the 125 uncopied files that the TUI pulls in transitively (16,686 lines in `vibe/cli/textual_ui`, 9,931 elsewhere).

**These files do not parse on Python < 3.12** (`type X = …` statements or PEP 695 generics):
- `vibe/cli/textual_ui/app.py` L580, L583, L584 (`type AppServerStarter = …` and two more)
- `vibe/cli/textual_ui/widgets/rewind_app.py` L32
- `vibe/cli/textual_ui/widgets/session_picker.py` L31
- `vibe/cli/textual_ui/screens/config/_common.py` L300
- `vibe/cli/textual_ui/widgets/vibe_code_project/picker.py` L103
- `vibe/cli/audio_request_metadata.py` L7
- `vibe/app_server/session.py` L125; `client.py` L29 and L341 (generic def); `_session_resources.py` L85; `_streaming.py` L11, L16, L24, L55 (generic defs)

**3.11+ stdlib names** (OVAT's `ovat/tui/vibe/_compat.py` already shims `StrEnum`, `UTC`, `Self`, `override` and `assert_never`; uncopied files need their imports redirected to it):
- `enum.StrEnum`: app.py L8; notifications/ports/notification_port.py; screens/config/config_screen.py; widgets/connector_auth_app.py; widgets/mcp_oauth_app.py; widgets/rewind_app.py; widgets/vibe_code_project/picker.py; word_selection.py; audio_player/audio_player_port.py; narrator_manager/narrator_manager_port.py; update_notifier/ports/update_gateway.py; observability/sentry.py; utils/api_keys.py
- `datetime.UTC`: widgets/session_picker.py L5; vscode_extension_promo/__init__.py L4
- `typing.assert_never`: app_server/_turn_input.py L4
- `tomllib`: utils/cache_store.py (used for the greeting and untrusted-config caches); update_notifier/adapters/uv_tool_update_gateway.py. On 3.10 you need `tomli`, which is **not installed** in OVAT's venv.
- `datetime.fromisoformat`: widgets/session_picker.py L45. It replaces `Z` first, but 3.10 still rejects other ISO forms (for example fractional seconds that are not 3 or 6 digits); test with real timestamps.

**Behavioral trap: builtin `TimeoutError` vs `asyncio.TimeoutError`.** Before 3.11, `asyncio.wait_for` raises `asyncio.TimeoutError`, which is **not** the builtin, so these clauses would not catch it:
- app.py L3872 `with suppress(TimeoutError): await asyncio.wait_for(...)` (interrupt waiting for a turn to surface)
- app.py L5697 `except TimeoutError:` in `_await_interrupt_settled`
- voice_manager/voice_manager.py L149; update_notifier/update.py L243; update_notifier/adapters/uv_tool_update_gateway.py L129

Use `except (TimeoutError, asyncio.TimeoutError)`.

No uses found of `itertools.batched`, `Path.walk`, `asyncio.TaskGroup`, `asyncio.timeout`, `ExceptionGroup`/`except*` or `tomllib` outside the two files above. (`match` statements, `X | Y` unions and `dataclass(slots=True)` are fine on 3.10.)

Already-copied files were not re-audited for this report. A quick grep found no remaining `TimeoutError`, `tomllib` or `from datetime import UTC` in `ovat/tui/vibe`, and `tool_widgets.py` already has its PEP 695 class rewritten with `Generic[TArgs]`.

---

## 6. Third-party packages used by the uncopied files

Tested with `/home/user/ovat/.venv/bin/python` (Python 3.13.16; Textual 8.2.8, which is also Vibe's pin and requires Python >= 3.9).

| Package | Used by | Importable in OVAT venv? |
|---|---|---|
| textual (`_xterm_parser`, `selection`, `cache`, `content`, `visual`, `scroll_view`, `strip`) | many | **yes** (8.2.8). `terminal_input_filter.py` and `word_selection.py` use private Textual APIs (`textual._xterm_parser`, `textual._context`), so keep the 8.2.8 pin. |
| rich | many | yes |
| pydantic | app_server/session.py, turn_summary/port.py, uv gateway | yes (2.14.1) |
| httpx | update_notifier GitHub and PyPI gateways, utils/http.py | yes |
| packaging | update_notifier | yes |
| certifi, truststore | utils/http.py | yes |
| keyring | utils/keyring.py | yes |
| pyperclip | cli/clipboard.py | yes |
| typing_extensions | (compat) | yes |
| **humanize** | widgets/feedback_bar.py (`naturaldelta`), chat_input/paste_image.py (`naturalsize`) | **NO** (trivial to replace) |
| **miniaudio** | audio_recorder, audio_player (voice, narrator) | **NO** |
| **mistralai** (`client`, `client.models`, `extra.realtime`) | transcribe/mistral_transcribe_client.py, tts/mistral_tts_client.py | **NO** |
| **sentry_sdk** | observability/sentry.py | **NO** |
| **tomli_w** | utils/cache_store.py | **NO** |
| **tomli** (3.10 fallback for tomllib) | needed by cache_store and uv gateway on 3.10 | **NO** |

---

## 7. Backend surface the TUI uses (what an OVAT adapter must provide)

**`AppServerSession` methods and properties used by app.py:**
- Properties: `session_id`, `cwd`, `state` (`.session.harness`, `.session.parent_session_id`, `.turns`), `history`, `turn_queue`, `child_sessions`, `turn_active`, `active_turn_id`.
- Methods: `exit_summary()`, `events()` (async stream), `act(...)` (used for retry and injected turns), `enqueue`, `replace_queued_turn`, `remove_queued_turn`, `resume_turn_queue`, `steer_queued_turn`, `refresh_state`, `inject_user_context`, `interrupt`, `respond_to_callback`, `resume(session_id)`, `compact`, `clear_history`, `begin_close`, `close`.

**Events consumed:** `TurnStarted`, `TurnCompleted`, `HistoryEntryAdded`, `HistoryEntryUpdated` (JSON-patch append), `SessionUpdated`, `SessionSnapshot`, `SessionContextCleared`, `ChildSessionUpdated`, `TurnQueueUpdated`, `StatsUpdated`, `CallbackRequested`, `ServerWarning`, `ServerError`. These models live in `ovat/tui/vibe/app_server/events.py` (copied).

**Resources used, by RPC method:**

| Resource | RPC methods |
|---|---|
| `config` | `current`, `subscribe`, `reload` (`config/reload`), `update`, `write` (`config/write`), `write_model`, `set_thinking`, `read_fields`, `read_proxy`, `update_proxy` |
| `runtime` | `ready`, `wait_until_ready`, `stats`, `context_window`, `issues`, `mcp`, `connectors`, `skills`, `get_skill`, `tools`, `has_tool`, `hooks_count`, `custom_skills_count`, `bypass_tool_permissions`, `experimental_harness`, `session_log`, `session_init_duration_ms`, `read_logs` |
| `agents` | `active`, `all`, `next`, `switch`, `set_installed` |
| `sessions` | `list`, `resolve_continue_session`, `get_session_history`, `list_history`, `history_before_cursor`, `load_before`, `delete`, `rename`, `fork`, `read_log`, `rewind_has_file_changes`, `rewind` |
| `workspace` | `prepare_prompt`, `untrusted_config_dirs` (plus `trust_status`/`decide_trust` on the host) |
| `shell` | `run` |
| `mcp` | `read`, `state`, `refresh`, `refresh_connectors`, `toggle`, `add`, `login`, `logout`, `connector_auth_url`, `refresh_connector` |
| `plugins` | `read`, `reload` |
| `skills` | `read_installed`, `catalog`, `updates`, `detail`, `versions`, `import_skill`, `set_version`, `set_latest`, `set_alias`, `remove`, `set_enabled` |
| `loops` | `list`, `create`, `delete`, `clear` |
| `provider_auth` | `read` |
| `identity` | `read`, `current` |
| `account` | `read`, `current` |
| `vibe_code` | `open_projects`, `load_more`, `create`, `select_project`, `unlink`, `cancel_picker`, `recover_stale_link`, `teleport`, `respond_to_push` |
| `narration` | `summarize` |
| `feedback` | `should_show`, `record` |
| `telemetry` | `record` |

`resources.refresh()` is also called. `review` is never used.

**Mapping to generic capabilities**, ordered from most to least needed:
1. Streaming turn events (text, reasoning, tool calls with typed I/O).
2. Interrupt.
3. Approval callbacks with session and permanent grants.
4. Ask-user questions.
5. Token stats and context window.
6. List, switch and cycle agent modes.
7. List and switch model; thinking levels.
8. Turn queue and mid-turn steering.
9. Saved sessions: list, preview history, resume, delete, rename, fork, continue-latest.
10. Clear and new session.
11. Compact.
12. Rewind with file checkpoints.
13. Paged history.
14. Prompt preparation (@-mentions, images).
15. User shell exec with streaming.
16. Layered config read/write/reload.
17. MCP server list, toggle, add and OAuth.
18. Skills list.
19. Plugin catalog.
20. Scheduled loops.
21. Subagent child sessions.
22. Todo tool events.
23. Hooks events.
24. Plan-mode notices.
25. Diagnostics logs.
26. Folder trust.
27. Proxy settings.

---

## 8. Brand and trademark items OVAT must replace

Paths are Vibe paths. "OVAT copy" means the copied file under `ovat/tui/vibe/` still contains the item (checked by grep).

| Item | Location | OVAT copy status |
|---|---|---|
| "Mistral Vibe" banner brand text | widgets/banner/banner.py L80 | **still present** in OVAT copy |
| Cat mascot "petit_chat" (`PetitChat`, `CatVariant.LECHAT`/`LECHONK`; "Le Chonk" variant picked for "mistral large"/"ml4" models), `.petit-chat`/`.lechonk` CSS | widgets/banner/petit_chat.py 692 lines (L546–692 logic, rest is art); braille_renderer.py; app.tcss ~L1735–1850 | **still present** (copied verbatim) |
| `MistralColors` enum (`#E10500`, `#FA500F`, `#FF8205`, `#FFAF00`, `#FFD800`) used for the loading gradient | textual_ui/constants.py L11; widgets/loading.py L42–46 | OVAT recolored the values to blues but kept the names `MistralColors`/`ORANGE*` |
| `$mistral_orange: #FF8205` CSS variable (used in about 10 rules) | app.tcss L1, L80, L299–314, L496, L508, L1765, L1805, L1843 | OVAT recolored to `#00C7FD`; the variable name is unchanged |
| `[bold orange]` narrator animation | widgets/narrator_status.py L68, L74 | not copied |
| Loading easter eggs: "Seeding Mistral weights", "Vibing", "Sending good vibes", "Petting le chat", "Petting le chat noir", "Oui oui baguette", "Reading Proust", "Synthèse" | widgets/loading.py L55–75 | OVAT replaced the main list; **"Petting le chat noir" (Halloween list) is still present** |
| "Mistral Vibe has been suspended. Run fg to bring Mistral Vibe back." | app.py L6583 | not copied |
| Tab default title "Vibe" | app.py L758; textual_notification_adapter.py L104 | not copied |
| Process name "Vibe CLI" | cli/_process_title.py | not copied |
| "Interrupted | What should Vibe do instead?" | widgets/messages.py L540 | **still present** (OVAT messages.py L542) |
| "Ask Vibe to help replace yours (…)" (custom tools deprecation) | messages.py L522 | **still present** |
| `VSCODE_EXTENSION_URI = "vscode:extension/mistralai.mistral-vibe-code"` and the promo message | messages.py L471 | **still present** |
| "How's your session with Vibe?" | widgets/feedback_bar.py L33 | not copied |
| "Once authenticated in your browser, return to Vibe" | widgets/mcp_oauth_app.py L211 | not copied |
| "Set up credentials in the Mistral dashboard…" | widgets/mcp_app.py L510 | not copied |
| "Sign in to Mistral to browse shared skills"; ".vibe/skills.toml" | widgets/skills_browser.py L593, L876 | not copied |
| "No Mistral account information is available." (/whoami) | app.py L4292 | not copied |
| `/whoami` description "Display the Mistral signed-in user…"; `/teleport` "Vibe Code Web"; `/remote-project` "Vibe Code Web project"; `/leanstall` "(leanstral)"; `/branch` "Resume the copy with `vibe --resume <id>`" | cli/commands.py | **still present** (commands.py copied) |
| "To resume it later, run: `vibe --resume …`" | app.py L4932 | not copied |
| "…or ask Vibe to do it." (untrusted config warning) | app.py L6278 | not copied |
| "Vibe Code" strings (teleport and project picker) | app.py L2488–2609, L3617–3683, L5165; teleport_message.py; vibe_code_project/* | not copied (CLOUD; drop) |
| Data retention text and the chat.mistral.ai URL | utils/data_retention.py | not copied (drop) |
| "This session was created by a newer version of Vibe…" | textual_ui/_resume_errors.py | not copied |
| Update check package name "mistral-vibe" | app.py L6664 | not copied |
| Env vars `VIBE_TYPING_GRACE_PERIOD_MS`, `VIBE_WHATS_NEW_FILE`; temp prefixes `vibe_` (external_editor.py L26, **copied**), `vibe-pasted-images` (paste_image.py) | various | external_editor prefix **still present** |
| TextArea theme name `vibe-chat` | chat_input/text_area.py L54 | **still present** (internal, harmless) |
| Telemetry event names `vibe.*`; Sentry tag `vibe_boundary` | app.py (many) | not copied |
| Logger names `"vibe"`, `"mistralai_vibe_local_harness"` | observability/logging.py L18, L29 | **still present** (internal) |
| `vibe/whats_new.md` content | repo root `vibe/whats_new.md` | not copied |

---

## 9. Notes and uncertainties

- Line numbers refer to commit 4ae5c59.
- Some features exist only in Vibe's "unified harness" (`experimental_harness`): /plugins, /reload-plugins, /todo with the pinned row and overlay, natural-language /loop, and the Model & Provider section of /status. Porting them makes sense only if OVAT has the matching capability.
- I did not run Vibe; behaviour descriptions come from reading the code.
- The onboarding, setup/auth and update-prompt apps were skimmed by file list and `cli.py` call sites only, not read line by line.
- "Copied" status was checked by path existence. OVAT's `app.tcss` differs from Vibe's (recolored). OVAT's `windowing/__init__.py` is empty, so `windowing/state.py`, `history_windowing.py` and `widgets/load_more.py` must be copied for #62.
