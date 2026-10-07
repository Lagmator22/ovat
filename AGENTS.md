# AGENTS.md: read this before touching anything

OVAT = OpenVINO Agentic Toolkit. GSoC 2026 project #18 (Intel/OpenVINO,
mentors Freddy Chiu and Ravi Panchumarthy). Owner: Gurman (GitHub
Lagmator22). Mission: "one YAML + one command." Turn tool-calling agent
boilerplate into `ovat run workflow.yml --input "..."`, backed by OVMS on
Intel AI PCs, with a local no-server path for dev machines. Published on
PyPI as `ovat` (pyproject says 1.0.9 at the time of writing), with an
optional Claude-Code-style TUI.

This file is the project's agent instructions for every coding agent.
Claude Code (v2.1.277+) reads it directly, but ONLY while no `CLAUDE.md` or
`CLAUDE.local.md` exists in the repo or above it. Do not create either one:
the moment one exists, this file silently stops loading.

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
                    └─> the reply carries tool_calls ─> run tool ─> loop
                        the reply carries none ─> answer
                        (trace in agent.last_trace; framework engines
                         set agent.last_failed instead)
```

The loop dispatches on the PAYLOAD (does the message carry tool calls), not
on `finish_reason`, because OVMS's NPU path documents `finish_reason`
as always `"stop"`.

## Map (each file, one line)

### config, agent, providers
- `ovat/config/workflow.py`: pydantic schema for workflow.yml. StrictModel
  base: unknown keys are ERRORS. A new field needs: schema + a PURPOSE entry
  in `scripts/gen_workflow_reference.py` + regenerate
  `docs/workflow_yaml_reference.md` + examples/ if relevant + a test.
  Also derives `tool_parser` from the model name (`_PARSER_BY_FAMILY`).
- `ovat/agent/loop.py`: the native tool-calling loop + run trace (Layer 7).
  Flags `empty_answer`, `undecoded_tool_call`, `truncated` in the trace.
- `ovat/agent/factory.py`: config -> wired agent. Tool registry
  (`BUILTIN_TOOL_SCHEMAS` + builders), `mcp_stdio` hookup, `build_llm`,
  `close_agent`.
- `ovat/agent/arg_models.py`: derives per-framework argument models from
  each tool's SCHEMA. Pydantic only, no framework imports. EVERY engine
  derives from here; a hand-kept second registry is how a tool ended up
  working on one engine and crashing on another.
- `ovat/agent/langchain_agent.py`: `agent.type: react`.
- `ovat/agent/llamaindex_agent.py`: `agent.type: llamaindex`. OpenAILike must
  have BOTH is_chat_model and is_function_calling_model set, or it calls an
  endpoint OVMS does not serve / silently degrades to plain chat. Passes
  `max_iterations` to `run()` (LlamaIndex's own default is 20).
- `ovat/agent/openai_agents_agent.py`: `agent.type: openai-agents`. Three
  things stop it phoning OpenAI instead of OVMS: an explicit AsyncOpenAI
  client on the /v3 base URL, OpenAIChatCompletionsModel (not the default
  Responses model), and set_tracing_disabled(True). Do not remove any. A
  made-up tool name (ModelBehaviorError) becomes a readable answer plus
  `last_failed`, not a crash.
- Both async framework engines own their event loop and REFUSE to run inside
  an existing one rather than nesting; do not "fix" that.
- `ovat/agent/session.py`: conversation memory + JSON save/load (TUI /save,
  /load, autosave).
- `ovat/agent/rag_chat.py`: local retrieve-then-answer, no tool calling;
  `history` (last 8 messages) and `on_token` streaming (return True from the
  callback to STOP, per openvino_genai's contract).
- `ovat/providers/base.py`: the ABCs (LLM/Embeddings/Retriever/VLM).
- `ovat/providers/backend.py`: `LLMBackend`, ONE description of the OVMS
  backend (url, model, timeout, sampling, `openai_kwargs()`, `extra_body()`)
  that all four engines read. Four copies of the temperature drifted once;
  new sampling knobs go here and nowhere else.
- `ovat/providers/llm_ovms.py`: OpenAI SDK -> OVMS /v3; returns `usage`;
  bounded by `model.request_timeout` (NEVER remove the timeout).
- `ovat/providers/llm_genai.py`: local openvino_genai LLM, no tool calls,
  streams via `on_token`. Text models get an `ov_genai.ChatHistory`, so the
  model's own chat template is applied; unified models keep the formatted
  string unless `enable_thinking` is set. `ovat chat` and TUI `/chat`.
- `ovat/providers/embeddings_genai.py` / `embeddings_ovms.py`: text->vectors.
- `ovat/providers/retriever_sqlitevec.py`: the on-disk vector store;
  `check_same_thread=False` (LangChain runs tools on a worker thread);
  `k = ?` not `LIMIT ?` (see History, SQLite < 3.38); close() idempotent.
- `ovat/providers/retriever_memory.py`: in-memory vector store, the second
  Retriever implementation (proves the ABC is the right shape).
- `ovat/providers/vlm_genai.py`: vision model, reached via `describe_image`.

### core
- `ovat/core/model_server.py`: OVMS lifecycle: build args, start (logs to
  file, pidfile, `ovms_env()` for PATH / LD_LIBRARY_PATH), wait_until_ready
  (a STALL budget, see Landmines), stop, `stop_from_pidfile`.
- `ovat/core/ovms_locator.py`: find the ovms binary: config `ovms_binary` ->
  `OVAT_OVMS` -> PATH -> known unzip folders. Windows installs are NEVER on
  PATH; this is why serve works anyway.
- `ovat/core/ovms_installer.py`: `ovat setup`. Picks and verifies the right
  OVMS archive (never `python_off`, which cannot tool-call), extracts it
  safely into a folder the locator searches. Pins `OVMS_VERSION = 2026.2.1`.
- `ovat/core/model_scout.py`: find/identify local model folders (llm, vlm,
  unified, whisper, embeddings) from file layout + config.json.
  `CHAT_KINDS` is the one list of kinds a chat can use.
- `ovat/core/device_manager.py`: CPU/GPU/NPU routing table (doctor, init).
- `ovat/core/model_manager.py`: thin wrapper over `ovms --pull/--list_models`.

### tools, rag, telemetry, top level
- `ovat/tools/search_docs.py`, `transcribe.py`, `describe_image.py`: builtin
  tools. Pattern: plain `*_impl()` (testable), co-located `SCHEMA` (THE
  contract, carries defaults), FastMCP wrapper + `mcp.run()` under
  `__main__` (standalone MCP server mode).
- `ovat/tools/fuzzy.py`: a misspelt file path is swapped for the closest
  existing file, BOUNDED (the named folder, or two levels under the cwd for
  a bare name) and ANNOUNCED (a note leads the tool result).
- `ovat/tools/mcp_client.py`: MCP stdio CLIENT (official `mcp` SDK). One
  event-loop thread per server; connect/serve/unwind in ONE manager
  coroutine (anyio cancel scopes must enter/exit in the same task). The SDK
  passes the server only HOME/LOGNAME/PATH/SHELL/TERM/USER; `tools[].env`
  adds more.
- `ovat/rag/indexer.py`: chunk (+overlap) and index .txt/.md folders.
  `on_progress(done, total, path)` fires once per file, AFTER it is stored.
- `ovat/telemetry/base.py`: Layer 7 contracts, `TelemetrySource` (where a
  number comes from) and `TelemetrySink` (where it goes). N+M, not N*M.
- `ovat/telemetry/sources.py`: system, process memory, agent trace,
  `NPUSource` (Linux sysfs / Windows PDH `_WindowsNPUCounter`),
  `IntelHardwareSource` (Intel Unified Telemetry, `find_ut`), `OVMSLogSource`
  (KV cache from ovms.log). None of them raises.
- `ovat/telemetry/sinks.py`: `JSONFileSink` (keeps its first write `error`),
  `LiveBufferSink`, `FanOutSink`. None of them raises.
- `ovat/telemetry/collector.py`: ties sources to sinks on a clock; reports
  `unavailable` and live-but-silent `notes` separately.
- `ovat/bench.py`: `ovat bench`, one question through several engines, each
  in a FRESH PROCESS (`python -m ovat.bench --worker`). `--repeat N` folds N
  runs into a success rate with medians. A failing engine is a ROW, not a
  crash. `_PeakMemory` is reused by `ovat run --trace`.
- `ovat/text.py`: reading model output (`strip_thinking`,
  `looks_like_undecoded_tool_call`, `says_nothing`, `strip_code_fence`).
  Top level because both agent/ and cli/ need it and neither may import the
  other.

### CLI and TUI
- `ovat/cli/main.py`: all 11 typer commands: run, chat, index, init, models,
  setup, serve, doctor, tui, telemetry, bench. All printing via the ONE
  themed console (`rprint`). `_load_config()` is the ONLY place allowed to
  call `load_workflow`. `_brief_error()` keeps the actionable fragment of a
  failure for the bench table. "Error talking to OVMS" is printed only for
  an `openai.APIError`.
- `ovat/cli/ui.py`: PALETTE (single source of truth for ALL colors, CLI and
  TUI), console theme, `wordmark()`.
- `ovat/cli/diagnostics.py`: doctor's checks. Platform-aware (macOS gets
  "OVMS does not run here", not "not on PATH").
- TUI (the `[tui]` extra): `tui.py` (launcher + masthead), `shell.py`
  (subprocess layer, slash TEMPLATES, \r progress sampling, process-tree
  kill), `chat_screen.py` (in-process chat, streaming, sessions in
  `.ovat/sessions/`, `/engine` local <-> OVMS), `doctor_screen.py`,
  `telemetry_screen.py` (live numbers, Digits + TabbedContent),
  `widgets.py` (PasteInput, ChatInput, SelectableRichLog), `editing.py`
  (InputHistory + clipboard, no textual), `theme.py`, `commands.py`
  (palette providers).

### outside ovat/
- `scripts/gen_workflow_reference.py`: generates
  `docs/workflow_yaml_reference.md` from the schema. Never hand-edit the
  generated file; a test fails when it is stale.
- `docs/ARCHITECTURE.md` (layers 1-9, the design record the mentors asked
  for), `docs/BLOG.md`, `examples/` (minimal, rag, react, audio-multimodal,
  document-qa, document-qa-npu, plano).
- `.github/workflows/tests.yml` (CI matrix, see History) and `publish.yml`
  (PyPI, `workflow_dispatch` only).

## HARD RULES: breaking these is how you nuke the codebase

1. **Git.** `main` is the trunk and the full toolkit (CLI + TUI). Work on a
   feature branch, one branch per feature or fix group, and land it through
   a PR. Never commit to or push `main` directly. Never rebase, never
   force-push, never rewrite pushed history: a mistake in a pushed commit is
   undone by a NEW commit. Stacked branches merge bottom-up with merge
   commits (not squash), so the commits above keep their identity.
2. **Push only when the owner asks**, and only feature branches. The AI PC
   has no GitHub credential and no `gh`: commit there, report the hashes,
   and the owner pushes.
3. **Stage files by name.** Never `git add -A`, `git add .` or a whole
   directory. The owner's tree holds untracked files of his own (scratch
   scripts, downloaded samples, demo tooling); a broad add committed two of
   them in Oct 2026. And remember git DELETES an untracked-on-target file
   from disk when you check out a commit that no longer tracks it: back it
   up first.
4. **Isolation contract**: the CLI must fully work with NO TUI installed.
   `textual`/`pyfiglet` live in the `[tui]` extra only. Module-level
   `import textual` is allowed ONLY in: `tui.py`, `chat_screen.py`,
   `doctor_screen.py`, `telemetry_screen.py`, `widgets.py`, `theme.py`,
   `commands.py`. `tests/test_tui_isolation.py` enforces this. The same
   applies to the framework engines: importing the CLI must pull in none of
   langchain, llama_index, agents, textual, pyfiglet.
5. **Single sources of truth**: colors -> `ui.PALETTE`; tool contracts -> the
   co-located `SCHEMA` dicts (every engine derives via `arg_models.py`);
   backend settings -> `providers/backend.LLMBackend`; config validity ->
   `workflow.py`; config LOADING -> `main._load_config`; model kinds a chat
   accepts -> `model_scout.CHAT_KINDS`. Never create a parallel copy:
   duplication is how the chat header said "engine: local" after a switch
   to OVMS, and how /chat refused the Qwen3.5 model `ovat chat` accepted.
6. **Tests gate everything.** Run `python -m pytest -q` with the venv's own
   interpreter (`.venv/bin/python` on macOS, `.\.venv\Scripts\python.exe` on
   the AI PC), never the system one. About 760 tests; must end green
   (`-n auto` works, pytest-xdist is in `[dev]`). Every fix ships with a
   test, and that test must FAIL with the fix backed out: verify it, do not
   assume. A test that passes against the broken code is worse than none,
   and that has happened here more than once. One logical change per
   commit, conventional message, body explains WHY.
7. **Errors are for users**: tool/agent failures become readable strings the
   model (or human) can act on, never bare tracebacks in the CLI/TUI path. A
   run that FAILED must still exit non-zero, write its trace, and clean up.
8. **Public writing** (commits, PRs, docs, README) is under the owner's name:
   no em dashes, no padded AI-voice paragraphs, and never a
   `Co-Authored-By: Claude` trailer (it puts "claude" in GitHub's
   Contributors panel and is near-impossible to remove).
9. **Verify against the primary source before you claim or build.** Read the
   official docs or the project's own repository (OVMS, OpenVINO,
   openvino_genai, Textual, the MCP SDK, the Agents SDK, plano) BEFORE
   writing a claim into a doc or writing code against someone else's
   behaviour. Not memory, not inference from a related page, not a
   plausible-sounding default. Every one of these shipped because that step
   was skipped:
   - `tool_parser: auto` was assumed to pick a parser. On 2026.2.1 it picks
     none.
   - "NPU cannot do tool calling" was wrong twice. OVMS documents NPU
     serving WITH tool calling, on this project's own silicon.
   - "agents are 90% plumbing" was a headline number with no source.
   - `ovms_cache_size_gb` was typed float; OVMS's uint64 parser refused
     every value. One line of OVMS's option reference would have caught it.
   - "Intel UT continuous mode prints nothing" was inference. It prints
     text; it was rejecting the flag combination OVAT passed.
   - a cursor bug was diagnosed twice from reading Textual's source and was
     wrong both times; printing what the widget held found it in one line.

   When a fact cannot be sourced, write "not verified" in the text. A
   measurement whose INTERPRETATION is a guess must say which half is which.
   Prefer the upstream repo over a docs site when they might differ: the
   docs describe a release, the source describes what runs.
10. **Explain in plain language.** The owner is a strong C++ dev and a
    Python beginner; C++ analogies land well. He decides who writes the
    code: when he asks for fixes, write them; when he asks to learn, guide.
11. **Rollback points**: tags `v0.2.0-w5-6-midterm`, `pre-tui-repair`,
    `v0.2.0-w7-8-complete`, and the release tags `v1.0.0` to `v1.0.9`. Cut a
    new tag before anything risky.

## Landmines: each of these cost a whole session once

Do not "improve" any of these without reading the reason first.

### OVMS and models
- **`tool_parser: auto` decoded NOTHING for Qwen3.5 on OVMS 2026.2.1.**
  OVMS documents picking a parser from the chat template when the flag is
  absent. Measured on live OVMS: it picked none and returned the call as
  plain text, `finish_reason: "stop"`, zero tool calls, the raw
  `<tool_call><function=...>` markup printed as the answer. `qwen3coder`
  for Qwen3.5/3.6/Qwen3-Coder, `hermes3` for Qwen3; the other families in
  `_PARSER_BY_FAMILY` are sourced from OVMS's own demos, not measured. NAME
  one. The failure is silent: the agent answers fluently and never calls a
  tool, so check for a CITATION or a tool_calls count in the trace. OVMS's
  automatic detection was rewritten in 2026.3.0 (PR #4312), AFTER this
  measurement: re-measure on a current OVMS before trusting either claim.
- **A pipeline that CONSTRUCTS is not the right pipeline.** On a unified
  Qwen3.5 export `openvino_genai.LLMPipeline` builds (24.6 s) and dies on
  the first `generate()` with "Port for tensor name input_ids was not
  found". `VLMPipeline` works. Pick from the export's file layout
  (`model_scout.identify_model`), never from a successful constructor.
- **One genai pipeline, one input type.** openvino_genai refuses to switch a
  pipeline between a string and a ChatHistory ("Chat doesn't support
  switching between input types"). Pick one per pipeline.
- **Never time-box a model download.** A first run needs ~185 s just to
  fetch Qwen3.5-4B, unbounded on a slow link. `wait_until_ready` uses a
  STALL budget: the clock resets whenever the log or the model repository
  grows.
- **Nothing capped a generation until 2026-08-12.** A model that never
  emits a stop token generated until the CLIENT gave up (1200 s for run,
  600 s for a bench worker). Greedy decoding makes it likelier. Its
  signature in `ovms.log`: the KV cache climbing for the whole run, 0.62 ->
  3.6 GB in ~9 minutes on ONE request.
  `model.max_tokens` defaults to 4096 now (real answers: 458-929 completion
  tokens). The KV cache climbing monotonically through one request is the
  SYMPTOM, not the cause. An earlier session read that arrow backwards and
  blamed llamaindex for "inheriting a full cache because it runs third"; the
  run that settled it had native fail FIRST while llamaindex passed in 33 s. A
  reply cut at the ceiling mid-markup looks exactly like an undecoded tool
  call: the loop tells them apart on `finish_reason: "length"` and reports
  `truncated`. Do not let that error blame the parser or the cache.
- **`ovms_cache_size_gb` is an int.** OVMS's `cache_size` is uint64 and its
  parser rejects "1.0"; it exits before opening its log, so `serve` can
  only say "exited without becoming ready". Read OVMS's stdout.
- **Tool calls that stop decoding on a LONG-LIVED OVMS: UNREPRODUCED.**
  2026-08-12: 4/4 runs failed `undecoded_tool_call` on a server that had
  served a bench and long runs (KV cache later 5.8 GB, `ovms.exe` 10.6 GB
  RSS); 17/17 passed on a fresh one, same model and parser. A percentage of the KV cache only means "full" when the log says
  `Cache type: static` (only with `--cache_size`): unset, OVMS grows the
  cache dynamically (measured 248.5 MB -> 5.6 GB over one session) and sat
  at or near 100% of the current allocation for 61.6% of readings
  (2449/3975), so "it failed at 98-100%" is a base rate, not evidence.
  Controlled version, `test_ovms_react_calls_a_tool_through_langchain`:

  | cache | result | latency |
  | --- | --- | --- |
  | static 1 GB, 13.9% | pass | 16 s |
  | static 1 GB, 100% | **FAIL** (`tool_calls: 0`) | 151 s |
  | static 1 GB, 100% | pass | 146 s |
  | static 8 GB, 1.4-1.7% | pass, pass, pass | 15 s, 18 s, 13 s |

  So: a full cache costs ~10x latency reproducibly (OVMS's documented
  preemption-and-recompute); breaking tool decoding is NOT shown at n=3. Restart OVMS before a
  demo; if it recurs, capture `--trace` AND `ovat telemetry` KV figures in
  the same window.
- **Same prompt, temperature 0, different outcomes.** The AI PC measured a
  qwen3coder engine calling its tool 1 run in 5 and another 5 in 5. One run
  proves nothing: use `ovat bench --repeat N`. The Qwen3.5 model card warns
  greedy decoding causes endless repetition, which is OVAT's default
  `temperature: 0.0`.
- **Knobs that exist and are NOT yet measured (2026-10-07).** All OFF by
  default: `model.ovms_tool_guided_generation` (OVMS
  `--enable_tool_guided_generation`, in 2026.2.1), `model.enable_thinking`
  (OVMS `chat_template_kwargs`; genai `set_extra_context`), `top_p`,
  `top_k`, `min_p`, `presence_penalty`, `seed`. Decide defaults from
  `ovat bench --repeat N` on the AI PC, not from the model card.

### NPU
- **OVMS compiles an LLM for NPU only from a channel-wise symmetric INT4
  export** (`-int4-cw-ov`). `OpenVINO/Qwen3-8B-int4-cw-ov` compiled in 36 s
  and served a real tool call (`examples/document-qa-npu.yml`); stock
  `-int4-ov` dies with `0x78000004 - [NPU_VCL]`. The compiler's own reason
  was `StopLocationVerifierPass ... Found 8 duplicated names` and OVMS
  loaded it as a VLM servable, so group quantisation as the CAUSE is
  unverified.
- `finish_reason` was `"tool_calls"` on NPU on this build, not the `"stop"`
  OVMS's demo documents. Dispatching on the payload is right either way.
- NPU is a Stateful servable: `cache_size`, `dynamic_split_fuse`,
  `max_num_batched_tokens`, `enable_prefix_caching` are IGNORED. Instead a
  STATIC total-sequence cap from `MAX_PROMPT_LEN`/`MIN_RESPONSE_LEN`: pulled
  with `--max_prompt_len 2000`, generation stopped at exactly 2129 total
  tokens, mid-sentence, `finish_reason: "unknown"`. A real way to hand the
  parser half a `<tool_call>`.
- **Windows NPU utilisation**: `Get-Counter -ListSet *NPU*` is a dead end
  (it matches "I-npu-t"). Use `\GPU Engine(*)\Utilization Percentage` on
  the Intel(R) AI Boost adapter. The GPU's `engtype_neural` is NOT the NPU:
  it read 99.8% with the NPU idle. Implemented as `_WindowsNPUCounter`,
  verified with both controls (NPU load 93.5%; GPU load, NPU idle 0.0%).

### Measurement
- **Absent is not zero.** Tokens, peak RSS, anything unknown stays `None`
  and renders as a dash. A zero reads as "used no tokens".
- **Peak RSS must be SAMPLED during a run**; afterwards Python has freed it.
- **`ovat run --trace` peak RSS measures the CLIENT.** With OVMS serving,
  the weights live in `ovms.exe`; the trace says ~0.45 GB. Qwen3.5-4B
  measured from the OVMS process: 4.3 GB steady, 6.5 GB peak. On the local
  `ovat chat` path the trace IS the model.
- **Bench engines run in separate processes.** In one process each engine
  inherited its predecessors' memory: native read 465.8 MB first and
  1155.6 MB last. A benchmark that changes its answer when you reorder the
  inputs is not measuring the inputs.
- **The trace's engine name comes from the config**, never a literal.
- **Intel UT (ut-tool-ext-v0.2.0-beta1.1) rejects `--continuous` with
  `--output`** and exits within a second; the hardware source was dead from
  dade76f until Oct 2026. It now runs with its cwd set to a scratch folder.
  `--continuous` alone also STOPS after one second (`-t` defaults to 1);
  OVAT passes `-t 86400`. Continuous mode prints `Metric: NAME | ... |
  Value: N unit | Timestamp: T | Duration: D` lines, parsed by
  `_parse_metric_line` (tests use captured lines verbatim). Duration is
  nanoseconds by MEASUREMENT, not from UT's docs. Only socwatch produces
  data in this mode; names stay UT's own (vccgt_power_w, not gpu_power_w)
  because UT does not document which rail powers what. ut also sometimes
  exits right after its first line, intermittently (2 of ~11 starts on
  2026-10-07, both with OVMS busy), cause not determined.

### TUI
- **`#masthead` height stays 17.** At 18 the TUI HANGS at 80x24.
- **`#brand-panel` is a COLUMN COUNT (42), never a percentage.** FIGlet art
  wider than the panel WRAPS and shears.
- **No Tooltips anywhere.** Tried, removed, tests assert none exist.
- **No `priority=True` bindings on scrolling keys.** They stole keys from
  the slash menu and modals.
- **Never mix `stream.write()` and `response.update()` on one Markdown
  widget.** MarkdownStream keeps its own record; once the text reshapes,
  retire the stream for that turn. Tags arrive SPLIT ACROSS TOKENS (`<th`
  then `ink>`).
- **Textual already binds `ctrl+c,super+c` to `screen.copy_text`** and
  provides `ctrl+p`; the app's copy branch in `action_cancel_or_quit` is
  what makes Ctrl-C copy. Do not declare a second `ctrl+p`.
- **RichLog cannot extract selected text.** It is a line-API widget, so
  Textual's generic `get_selection` returns None. Use `SelectableRichLog`.
- **Esc/Ctrl-C must end the whole process tree on Windows**, not just the
  direct child (`shell.py`); otherwise OVMS children outlive the command.

### Model discovery
- **`find_models` walks TWO levels.** `ovms --pull` lays models out by org
  (`models/OpenVINO/Qwen3-8B-int4-ov`).
- **Unified models** (Qwen3.5: one export that is both LLM and VLM) are a
  third kind answering both filters. On disk they look vision-only, so
  config.json is read BEFORE the layout is judged.

## Environment variables

Runtime: `OVAT_OVMS` (ovms binary/folder) · `OVAT_MODELS` (model discovery
roots, pathsep-separated) · `OVAT_VLM_MODEL` / `OVAT_WHISPER_MODEL` (tool
model dirs) · `OVAT_VLM_DEVICE` / `OVAT_WHISPER_DEVICE` (override the
routing table) · `OVAT_UT` (Intel UT folder or binary) · `OVAT_TUI=1` (set
by the TUI in children; bare `ovat` prints a hint instead of recursing).

Tests: `OVAT_TEST_MODELS_DIR` / `OVAT_TEST_IMAGE` (integration) ·
`OVAT_OVMS_URL` / `OVAT_OVMS_MODEL` (`test_ovms_live.py` target).

## Platform truth (answer users honestly)

- **macOS**: dev + tests + `ovat chat` + TUI `/chat` + `index`/`doctor`/
  `init` work (openvino_genai runs natively, CPU). No native OVMS, so no
  `serve`, `setup`, `models` or agentic `run`. OVMS's official amd64 Docker
  image runs under Rosetta (verified 2026-07-11, served bge-small): fine
  for the `live` tests with small models, slow for 8B. There is no arm64
  OVMS build.
- **AI PC / Windows / Linux**: everything. The owner's box:
  `C:\Users\devcloud\ovat`, OVMS at `C:\Users\devcloud\ovms_windows`,
  models under `C:\Users\devcloud\models\OpenVINO\`, reached over SSH from
  the Mac. Windows 11, Core Ultra (LunarLake), Arc 140V GPU + NPU.
- Verified for 1.0.0: Ubuntu 22.04.5 / SQLite 3.37.2 as uid 1000, Windows
  11 GPU + NPU, macOS as dev only. GPU/NPU on LINUX is untested
  (hardware-blocked: WSL2 has no /dev/dri, CI runners have no Arc or NPU)
  and stays marked untested in the README.

## Test suite conventions

`tests/conftest.py` has FakeLLMProvider/make_tool_call/reply and
`py_command()` (cross-platform). Markers: `live` (needs running OVMS), `rag`
(needs bge-small on disk); both auto-skip. Optional frameworks skip via
`pytest.importorskip` ("textual", "llama_index.core", "agents").
`test_docs.py` checks that docs and the generated reference do not drift.

TUI tests use Textual's headless Pilot via `asyncio.run` (no
pytest-asyncio). Mouse selection must be driven through
`Screen._forward_event`, NOT `post_message`.

Heavy seams for mocking: `chat_screen._build_components`,
`chat_screen._build_engine`, `cli_main.build_agent`,
`diagnostics.run_checks`, `model_server.subprocess.Popen`,
`bench.benchmark_engine`. Use `monkeypatch`, never a bare attribute
assignment: one leaked into other tests once.

Tests that scan the disk (`model_scout`, `fuzzy`) must isolate with
`monkeypatch.chdir` (and a fake HOME), or they describe the developer's
machine instead of the code.

## Open work (2026-10-07)

On the AI PC, measurement first:
- `ovat bench --repeat 5` A/B for each unmeasured knob above, then decide
  defaults.
- Intel UT: find why ut sometimes exits after its first line (see
  Measurement landmines).
- `enable_thinking: false` on a unified Qwen3.5 through local `/chat`.
- Upgrade the pinned OVMS (2026.2.1 -> current, 2026.4.1 at the time of
  writing) and re-measure `tool_parser: auto`.

Code and docs:
- Qwen2-VL produces "!!!!" through the local genai path (pre-existing,
  unexplained).
- GPU utilisation without Intel UT (Ravi's ask, 2026-08-21); the NPU
  counter is the sibling to build it on.
- A docs / codebase-navigation site and an API reference.
- Stress tests. GPU/NPU on Linux (hardware-blocked, above).

Scoped OUT by the owner, so do not re-propose: A2A orchestration (Layer 6),
OVMS Docker integration tests, and any audit of or comparison against
another vendor's agent toolkit. "Use every Textual widget" is not a goal:
Sparkline, Tree and MODES were tried or rejected; Digits and TabbedContent
earned their place on the telemetry page.

## History (why things are the way they are)

- **Midterm (2026-07-01, `v0.2.0-w5-6-midterm`)**: core proven live on the
  AI PC: native loop + RAG citations, transcribe, LangChain react, Qwen3-8B
  on GPU.
- **2026-07-03/04 repair**: strict config, request timeouts, serve pidfile
  + `--stop`, schema-derived tool args, one palette, `[tui]` extra +
  isolation tests, MCP stdio client, traces, DeviceManager, describe_image,
  ovms locator, model scout.
- **TUI finished 2026-07-28** and approved by the mentors; verified against
  live OVMS from the TUI itself.
- **W7-W8 (2026-07-29, `v0.2.0-w7-8-complete`)**: LlamaIndex and Agents SDK
  engines, `ovat bench`, `examples/document-qa.yml`. Same day, a telemetry
  audit fixed four defects that made numbers WRONG rather than missing.
- **Install repair (2026-08-01/03)**, after Ravi could not install from the
  README: unpublished package documented, missing `git clone`, doctor run
  before init, `optimum-cli` never installed, no OVMS steps, and
  `python_off` silently unable to tool-call. Also: small-model tiers (4.88
  GB -> 3.50 GB, plus a 0.91 GB tier), ARCHITECTURE.md, three worked
  examples.
- **AI PC clean-room run (2026-08-03)**: `tool_parser: auto`, first-run
  download timeout, locator missing the README's own folder, RAM measured
  on the wrong process, a stray `</think>` in every CLI answer.
- **1.0.0**: `ovat setup` installs OVMS in one command. Five bugs found by
  running it elsewhere, all "correct only on the machine that wrote it":
  non-root Linux extraction, an LD_LIBRARY_PATH branch never executed, a
  suite that could only pass on Windows, a 120 s timeout cutting off CPU
  servers (a cold run measured 1056 s), and RAG silently empty on SQLite <
  3.38. CI exists since then: ubuntu-22.04 + py3.10 is the oldest
  supported everything; jobs assert non-root, a framework-free BASE
  install, and that the built WHEEL runs. Publishing is
  `workflow_dispatch`: 0.9.11 and 0.9.12 never fired the release trigger.
- **2026-08-12**: Layer 7 telemetry (sources/sinks/collector, TUI page,
  `ovat telemetry`), the NPU serving work, `max_tokens`, the cache_size fix,
  the KV-cache experiments.
- **plano** (Ravi's ask, done): `examples/plano/` routes plano to OVMS (the
  /v1 vs /v3 path, the provider prefix, the missing-`id` bridge). Ports:
  plano 8000, bridge 8001, OVMS 8002.
- **Oct 2026 review round (PRs #30-#35)**: a whole-codebase review plus an
  AI PC verification pass. Framework engines now fail the run and the bench
  row when they fail; Intel UT starts again; local chat applies the model's
  chat template; sampling, thinking and tool-guided-generation knobs;
  `bench --repeat`; MCP `tools[].env`; per-family tool parsers; Windows
  process-tree kill; bounded, announced fuzzy paths; and a docs pass that
  removed claims the code contradicted.
