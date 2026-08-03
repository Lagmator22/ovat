# AGENTS.md: read this before touching anything

OVAT = OpenVINO Agentic Toolkit. GSoC 2026 project #18 (Intel/OpenVINO,
mentors Freddy Chiu & Ravi Panchumarthy). Owner: Gurman (GitHub Lagmator22).
Mission: "one YAML + one command." Turn tool-calling agent boilerplate into
`ovat run workflow.yml --input "..."`, backed by OVMS on Intel AI PCs, with a
local no-server path for dev machines. Think "NeMo Agent Toolkit, but for
OpenVINO", plus an optional Claude-Code-style TUI.

## The one flow to keep in your head

```
workflow.yml ──load_workflow()──> WorkflowConfig (pydantic, STRICT)
      └─build_agent()──> {LLM provider + tools + optional RAG retriever}
             └─> ONE of four engines, all exposing .run(text) -> text:
                   native         AgentLoop (loop.py), the only one that
                                  records per-turn tokens
                   react          LangChain
                   llamaindex     LlamaIndex FunctionAgent
                   openai-agents  OpenAI Agents SDK
                    └─> LLM says finish_reason: tool_calls ─> run tool ─> loop
                        LLM says stop ─> answer (trace in agent.last_trace)
```

## Map (each file, one line)

- `ovat/config/workflow.py`: pydantic schema for workflow.yml. StrictModel
  base: unknown keys are ERRORS. New fields need: schema + examples/ + README
  table row + a test.
- `ovat/agent/loop.py`: the native tool-calling loop + run trace (Layer 7).
- `ovat/agent/factory.py`: config → wired agent. Tool registry lives here
  (`BUILTIN_TOOL_SCHEMAS` + builders) and the `mcp_stdio` client hookup.
- `ovat/agent/arg_models.py`: derives per-framework argument models from each
  tool's SCHEMA. Pydantic only, no framework imports. EVERY engine derives
  from here; a hand-kept second registry is how a tool ends up working on one
  engine and crashing on another, which has happened once already.
- `ovat/agent/langchain_agent.py`: same job via LangChain (`agent.type:
  react`).
- `ovat/agent/llamaindex_agent.py`: `agent.type: llamaindex`. OpenAILike must
  have BOTH is_chat_model and is_function_calling_model set, or it calls an
  endpoint OVMS does not serve / silently degrades to plain chat.
- `ovat/agent/openai_agents_agent.py`: `agent.type: openai-agents`. Three
  things stop it phoning OpenAI instead of OVMS: an explicit AsyncOpenAI
  client on the /v3 base URL, OpenAIChatCompletionsModel (not the default
  Responses model), and set_tracing_disabled(True). Do not remove any.
- Both framework engines are async and own their event loop. They REFUSE to
  run inside an existing one rather than nesting; do not "fix" that.
- `ovat/agent/session.py`: conversation memory + JSON save/load (used by the
  TUI chat screen's /save /load and autosave).
- `ovat/agent/rag_chat.py`: local retrieve-then-answer (no tool calling);
  supports `history` (last 8 turns) and `on_token` streaming (return True
  from the callback to STOP generation, per openvino_genai's contract).
- `ovat/providers/base.py`: the ABCs (LLM/Embeddings/Retriever/VLM).
  Retriever has a default no-op `close()`.
- `ovat/providers/llm_ovms.py`: OpenAI SDK → OVMS /v3; returns `usage`;
  bounded by `model.request_timeout` (NEVER remove the timeout).
- `ovat/providers/llm_genai.py`: local openvino_genai LLM (no tool calls,
  streams via `on_token`). `ovat chat` and the TUI chat screen use this.
- `ovat/providers/embeddings_genai.py` / `embeddings_ovms.py`: text→vectors.
- `ovat/providers/retriever_sqlitevec.py`: vector store; `check_same_thread=
  False` (LangChain runs tools on a worker thread); close() is idempotent.
- `ovat/providers/vlm_genai.py`: Qwen2-VL vision; reached via the
  `describe_image` builtin tool.
- `ovat/core/model_server.py`: OVMS lifecycle: start (logs to file, pidfile,
  binary-dir prepended to child PATH), wait_until_ready, stop (terminate →
  kill), `stop_from_pidfile` for `ovat serve --stop`.
- `ovat/core/ovms_locator.py`: find the ovms binary: config `ovms_binary` →
  `OVAT_OVMS` env → PATH → known unzip folders. Windows installs are NEVER
  on PATH; this is why serve works anyway.
- `ovat/core/model_scout.py`: find/identify local OpenVINO model folders
  (llm/vlm/whisper/embeddings) from file layout + config.json. Powers chat
  auto-detection and the "that's a vision model" refusals.
- `ovat/core/device_manager.py`: CPU/GPU/NPU routing table; used by doctor
  ("Device routing" row) and `ovat init` (writes the detected device).
- `ovat/core/model_manager.py`: thin wrapper over `ovms --pull/--list_models`.
- `ovat/rag/indexer.py`: chunk (+overlap) and index .txt/.md folders.
  `on_progress(done, total, path)` fires once per file, AFTER it is stored.
- `ovat/bench.py`: `ovat bench` -- one question through several engines
  against one server, side by side. Peak RSS is SAMPLED on a thread (a single
  reading after the run misses the peak). A failing engine is a ROW, not a
  crash. `_PeakMemory` is reused by `ovat run --trace`.
- `ovat/tools/search_docs.py`, `transcribe.py`, `describe_image.py`: builtin
  tools. Pattern per tool: plain `*_impl()` (testable), co-located `SCHEMA`
  (THE contract: carries defaults; LangChain derives from it), FastMCP
  wrapper + `mcp.run()` under `__main__` (standalone MCP server mode).
- `ovat/tools/mcp_client.py`: MCP stdio CLIENT (official `mcp` SDK). One
  event-loop thread per server; connect/serve/unwind in ONE manager coroutine
  (anyio cancel scopes must enter/exit in the same task; do not "fix" this
  by closing from outside).
- `ovat/cli/main.py`: every typer command. All printing via the ONE themed
  console (`rprint = console.print`). `_load_config()` is the ONLY place
  allowed to call `load_workflow`: it turns a missing file / bad YAML /
  schema error into a sentence instead of a traceback. `_brief_error()`
  keeps the ACTIONABLE fragment of a failure for the bench table.
- `ovat/cli/ui.py`: PALETTE (single source of truth for ALL colors, CLI and
  TUI), theme, `wordmark()` (FIGlet + gradient, graceful without pyfiglet).
- `ovat/cli/diagnostics.py`: doctor's checks. Platform-aware: macOS gets
  "OVMS does not run here", not "not on PATH".
- TUI files (optional, installed via the `[tui]` extra): `ovat/cli/tui.py`
  (launcher + masthead), `ovat/cli/shell.py` (subprocess exec layer + slash
  TEMPLATES + \r progress sampling), `ovat/cli/chat_screen.py` (in-process
  chat: streaming, history, sessions in `.ovat/sessions/`, `/engine` to swap
  between the local genai model and OVMS-with-tools),
  `ovat/cli/doctor_screen.py` (in-app doctor, DataTable),
  `ovat/cli/widgets.py` (PasteInput, ChatInput, SelectableRichLog),
  `ovat/cli/editing.py` (InputHistory + system clipboard read, no textual),
  `ovat/cli/theme.py`, `ovat/cli/commands.py` (palette providers).

## HARD RULES: breaking these is how you nuke the codebase

1. **Branches**: everything lives on `main` now. The owner merged the TUI
   via PR #4 on 2026-07-04, so main IS the full toolkit (CLI + TUI). The
   old `week6-tui` branch is historical; do not develop on it. Never rebase
   or force-push anything. The TUI's separability is enforced by rule 3
   (the [tui] extra + isolation tests), not by branch topology anymore.
2. **Never push.** The owner pushes via his own tooling. Commit locally only.
3. **Isolation contract**: the CLI must fully work with NO TUI installed.
   `textual`/`pyfiglet` live in the `[tui]` extra only. Module-level
   `import textual` is allowed ONLY in: `tui.py`, `chat_screen.py`,
   `doctor_screen.py`, `widgets.py`, `theme.py`, `commands.py`.
   `tests/test_tui_isolation.py` enforces this: keep it passing.
   The same applies to the framework engines: importing the CLI must pull in
   none of langchain, llama_index, agents, textual, pyfiglet.
4. **Single sources of truth**: colors → `ui.PALETTE` only; tool contracts →
   the co-located `SCHEMA` dicts only (every engine derives via
   `arg_models.py`); config validity → `workflow.py` (strict); config LOADING
   → `main._load_config` only. Never create a parallel copy. Duplication is
   how two copies drift: the chat header said "engine: local" after a switch
   to OVMS for exactly this reason.
5. **Tests gate everything.** Run `python -m pytest -q` with the venv's own
   interpreter (`.venv/bin/python` on macOS, `.\.venv\Scripts\python.exe`
   on the AI PC), never the system one. ~522 tests; must end green. Every fix
   ships with a test, and that test must FAIL with the fix backed out: verify
   it, do not assume. A test that passes against the broken code is worse
   than none, and that has happened here more than once. One logical change
   per commit, conventional message, body explains WHY.
6. **Errors are for users**: tool/agent failures become readable strings the
   model (or human) can act on, never bare tracebacks in the CLI/TUI path.
7. Rollback tags exist: `v0.2.0-w5-6-midterm` (midterm), `pre-tui-repair`
   (TUI before the big repair), `v0.2.0-w7-8-complete` (all four engines +
   bench + sample agent). Cut a new tag before anything risky.
8. The owner is a strong C++ dev but a Python beginner: explain changes in
   plain language (C++ analogies land well). Default is that HE writes the
   code with guidance; only write code for him when he explicitly asks.

## Landmines: each of these cost a whole session once

- **`tool_parser: auto` decodes NOTHING for Qwen3.5.** OVMS documents that it
  picks a parser from the chat template when the flag is absent, so `auto`
  looked like the safe default. Measured on live OVMS: it selected no parser
  at all and returned the tool call as plain text, `finish_reason: "stop"`,
  zero tool calls, the raw `<tool_call><function=...>` markup printed as the
  answer. `qwen3coder` is the correct value for Qwen3.5, `hermes3` for Qwen3.
  NAME one. The failure is silent -- the agent answers fluently and simply
  never calls a tool -- so it survives every test that only checks for an
  answer. Check for a CITATION, or a tool_calls count in the trace.
- **A pipeline that CONSTRUCTS is not the right pipeline.** On a unified
  Qwen3.5 export, `openvino_genai.LLMPipeline` builds fine, taking 24.6s, and
  then dies on the first `generate()` with "Port for tensor name input_ids was
  not found". `VLMPipeline` is the one that works. So never infer support from
  a successful constructor; pick from the export's file layout instead (see
  `model_scout.identify_model`).
- **Never time-box a model download.** `wait_until_ready` had a fixed 120s cap
  and could not survive a first run, which needs ~185s just to fetch
  Qwen3.5-4B, and far longer on a slow link. Raising the number does not fix
  the shape: download time is unbounded. It is a STALL budget now -- the clock
  resets whenever the log or the model repository grows -- so a download that
  keeps moving is never interrupted while a wedged server still fails fast.
- **`ovat run --trace` peak RSS measures the CLIENT, not the model.** With
  OVMS serving, the weights live in `ovms.exe`; the trace reports ~0.45 GB and
  says nothing about the model. Measured truth for Qwen3.5-4B: 4.3 GB steady,
  6.5 GB peak, read from the OVMS process. The trace IS the right number on
  the local `ovat chat` path, where the model runs in-process.

Do not "improve" any of these without reading the reason first.

- **`#masthead` height stays 17.** At 18 the TUI HANGS at 80x24, a default
  terminal size. Measured. The brand stack is exactly 15 rows and the round
  border eats two, so it is an exact fit on purpose.
- **`#brand-panel` is a COLUMN COUNT (42), never a percentage.** It holds
  fixed-width FIGlet art. A mark wider than the panel does not clip, it WRAPS,
  which shears the glyphs.
- **No Tooltips anywhere.** Textual renders one as an unstyled block floating
  over content, anchored to the pointer, wrapping mid-phrase. On a
  full-screen app it covers the thing the user is reading. Tried, removed,
  and the tests now assert none exist.
- **No `priority=True` bindings on scrolling keys.** PageUp/PageDown/Home/End
  with priority stole keys from the slash menu and modals and broke the chat
  window. Scrolling already works.
- **Never mix `stream.write()` and `response.update()` on one Markdown
  widget.** MarkdownStream keeps its OWN record of what it wrote; update()
  does not tell it, so the next write appends to a stale buffer and the
  answer renders over itself with a broken border. Once the text reshapes,
  retire the stream for that turn. Tags arrive SPLIT ACROSS TOKENS (`<th` then
  `ink>`), so this fires near the start of most thinking-model answers.
- **The trace's engine name comes from the config**, never a literal. It was
  hardcoded "react" and started lying the moment a third engine existed.
- **Absent is not zero.** Token counts, peak RSS, anything unknown stays
  `None` and renders as a dash. A zero reads as "used no tokens" and a
  benchmark built on it is quietly wrong, which is the worst kind.
- **Peak RSS must be SAMPLED during a run.** A single reading afterwards
  misses the peak: Python has already freed the big allocations.
- **`find_models` walks TWO levels.** `ovms --pull` lays models out by org
  (`models/OpenVINO/Qwen3-8B-int4-ov`); a one-level scan reported "no local
  text LLM found" while the model sat right there.
- **Textual already binds `ctrl+c,super+c` to `screen.copy_text`.** The app's
  own priority ctrl+c shadows it on every screen, so the copy branch in
  `action_cancel_or_quit` is what makes Ctrl-C copy a selection. Do not
  declare a second `ctrl+p` either: Textual provides one and the Footer
  showed "^p Palette" at BOTH ends.
- **RichLog cannot extract selected text.** It is a line-API widget, so
  Textual's generic `get_selection` returns None and the clipboard comes back
  empty. `SelectableRichLog` implements it; use that class, not `RichLog`.

## Environment variables

`OVAT_OVMS` (ovms binary/folder) · `OVAT_MODELS` (model discovery roots,
pathsep-separated) · `OVAT_VLM_MODEL` / `OVAT_WHISPER_MODEL` (tool model
dirs) · `OVAT_TEST_MODELS_DIR` / `OVAT_TEST_IMAGE` (integration tests) ·
`OVAT_TUI=1` (set by the TUI in children; makes bare `ovat` print a hint
instead of recursing).

## Platform truth (answer users honestly)

- **macOS**: dev + tests + `ovat chat` + TUI `/chat` + `index`/`doctor`/
  `init` all work (openvino_genai runs natively, CPU). `ovat serve` has no
  native macOS path: no `serve`, no `models`, no agentic `run` via the
  binary. OVMS's official Docker image (amd64) DOES run under Rosetta on
  Apple Silicon (verified 2026-07-11: booted, served a real bge-small model,
  answered a REST inference request) - there is no arm64 image, so it is
  x86 emulation, fine for small models but slow for an 8B-class LLM. Useful
  for running the `live` OVMS tests on a Mac; the native genai pipeline
  stays the default for local LLM dev.
- **AI PC / Windows / Linux**: everything, including OVMS serving and
  tool-calling `run`. OVMS is usually NOT on PATH; the locator handles it.
  The owner's box: `C:\Users\devcloud\ovat`, OVMS at
  `C:\Users\devcloud\ovms_windows`, models under
  `C:\Users\devcloud\models\OpenVINO\`, reached over SSH from a Mac.
  Verified there 2026-07-28/29: all four engines answer through live OVMS on
  GPU, the native loop really calls tools (transcribe returned the JFK audio
  text; search_docs cites source paths), and `ovat bench` produces the
  four-engine comparison table.
- OVMS is x86-only; no arm64 build. On Apple Silicon it only runs via Docker
  under Rosetta emulation (see above), never bare metal.

## Test suite conventions

`tests/conftest.py` has FakeLLMProvider/make_tool_call/reply and
`py_command()` (cross-platform). Markers: `live` (needs running OVMS), `rag`
(needs bge-small on disk); both auto-skip. Optional frameworks skip via
`pytest.importorskip` ("textual", "llama_index.core", "agents").

TUI tests use Textual's headless Pilot via `asyncio.run` (no pytest-asyncio).
Mouse selection must be driven through `Screen._forward_event`, NOT
`post_message`: selection lives in `_forward_event`, and a probe that posts
events sees nothing and wrongly concludes the feature is broken.

Heavy seams for mocking: `chat_screen._build_components`,
`chat_screen._build_engine`, `cli_main.build_agent`, `diagnostics.run_checks`,
`model_server.subprocess.Popen`, `bench.benchmark_engine`. Use `monkeypatch`,
never a bare attribute assignment: a bare one leaked into other tests once.

Tests that scan the disk (`model_scout`) must isolate with `monkeypatch.chdir`
and a fake HOME, or they describe the developer's machine instead of the code.

## History (why things are the way they are)

- **Midterm (2026-07-01, tag v0.2.0-w5-6-midterm)**: core proven live on the
  AI PC: native loop + RAG citations, transcribe, LangChain react, serving
  Qwen3-8B on GPU.
- **2026-07-03/04 repair mega-session** (both branches, ~30 commits): full
  audit vs the proposal PDF, then: LICENSE/deps/version hygiene; sqlite
  close(); request timeouts; serve pidfile + --stop; loop edge cases; strict
  config; schema-derived LangChain args; one palette + themed console; TUI
  recursion guard (OVAT_TUI + TTY check), busy-gate race fix, Esc/Ctrl-C
  kill escalation, \r progress sampling, [tui] extra + isolation tests;
  native chat screen (streaming/history/sessions); MCP stdio client
  (`type: mcp_stdio` works, tested over a real wire); observability traces
  (`ovat run --trace`, psutil RSS); DeviceManager wired into doctor/init;
  describe_image tool; ovms locator; platform-aware doctor with the big
  FIGlet sign; model scout + chat auto-detection.
- **TUI finished 2026-07-28** and approved by the mentors. Message widgets,
  streaming, foldable reasoning, multi-line input, system-clipboard paste,
  Up/Down history, session picker, command palette, themes, DataTable doctor
  with severity sort and per-row copy, check_action greying, turn separators,
  indexing progress bar. Verified against live OVMS from the TUI itself
  (`/engine ovms` -> real tool calls). Widgets deliberately NOT used, so they
  are not re-proposed: Sparkline, Digits, Tree, TabbedContent, MODES.
  "Use every Textual widget" is not a goal; push back on it.
- **W7-W8 complete 2026-07-29** (tag `v0.2.0-w7-8-complete`): LlamaIndex and
  OpenAI Agents SDK engines, `ovat bench`, and `examples/document-qa.yml` as
  the Document-Q&A sample. Benchmarked on the AI PC: all four engines ok.
- **Telemetry audit 2026-07-29**: four defects fixed that made measurements
  WRONG rather than missing (engine mislabelled, peak RSS not a peak, unknown
  tokens reported as 0, bench hiding the install hint behind an exception
  class). Plus: a bad workflow path printed a raw traceback from all five
  commands that read one.
- **Install repair 2026-08-01/03**, after Ravi could not install from the
  README ("it only works on your laptop"). It was literally true. Six defects,
  each reproduced before fixing: `pip install ovat` was documented while the
  package was unpublished (PyPI 404); the quickstart never said `git clone`
  yet step 1 was `pip install -e "."`; `doctor` ran BEFORE `init` created the
  file it validates, so a new user's first command ended in red; `optimum-cli`
  was instructed by both the README and `ovat init`'s own output while nothing
  installed it; there were no OVMS install steps at all; and OVMS's `python_off`
  build cannot tool-call, so the wrong archive gives an agent that answers
  normally and silently never calls a tool. Delivered with it: a prerequisites
  section (GPU/NPU drivers, VC++ redist), small models (4.88 GB -> 3.50 GB,
  plus a 0.91 GB tier), `docs/ARCHITECTURE.md` linked from the README, three
  worked examples (`examples/rag`, `react`, `audio-multimodal`), and
  `docs/BLOG-OUTLINE.md`.
- **Unified multimodal models**: Qwen3.5 is ONE export that is both a text LLM
  and a VLM, so `model_scout` grew a third kind, `unified`, that answers to
  both the `llm` and `vlm` filters. On disk it is indistinguishable from a
  vision-only model, so config.json is read BEFORE the layout is judged.
- **AI PC clean-room run 2026-08-03**: a fresh clone driven exactly as the
  README says, with every OVAT_* variable cleared. Found five more, all
  invisible on a machine that already worked: `tool_parser: auto` decoding
  nothing, `serve` unable to survive a first-run download, the locator missing
  the very folder the README says to unpack OVAT into, RAM figures measured on
  the wrong process, and a stray `</think>` in every CLI answer. See Landmines.
- Published to PyPI as `ovat` (0.9.x), so `pip install ovat` finally works.
  `.github/workflows/publish.yml` handles releases.
- Still open: W9-W10 remainder (OpenTelemetry proper, stress tests, OVMS
  Docker integration tests), an API reference, CI for the test suite
  (deliberately postponed), and the mentor ask to strip/audit NVIDIA's NeMo
  Agent Toolkit for what OVAT should adopt (Ravi: critical). The plano ask is
  DONE: `examples/plano/` points it at OVMS, with the /v1-vs-/v3 path, the
  provider prefix and the missing-`id` bridge all solved and tested.
