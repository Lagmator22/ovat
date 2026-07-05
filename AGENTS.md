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
             └─> AgentLoop (native) or LangChainAgent (react)
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
- `ovat/agent/langchain_agent.py`: same job via LangChain (`agent.type:
  react`). Arg models are DERIVED from each tool's SCHEMA: never hand-write
  a second registry.
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
- `ovat/tools/search_docs.py`, `transcribe.py`, `describe_image.py`: builtin
  tools. Pattern per tool: plain `*_impl()` (testable), co-located `SCHEMA`
  (THE contract: carries defaults; LangChain derives from it), FastMCP
  wrapper + `mcp.run()` under `__main__` (standalone MCP server mode).
- `ovat/tools/mcp_client.py`: MCP stdio CLIENT (official `mcp` SDK). One
  event-loop thread per server; connect/serve/unwind in ONE manager coroutine
  (anyio cancel scopes must enter/exit in the same task; do not "fix" this
  by closing from outside).
- `ovat/cli/main.py`: every typer command. All printing via the ONE themed
  console (`rprint = console.print`).
- `ovat/cli/ui.py`: PALETTE (single source of truth for ALL colors, CLI and
  TUI), theme, `wordmark()` (FIGlet + gradient, graceful without pyfiglet).
- `ovat/cli/diagnostics.py`: doctor's checks. Platform-aware: macOS gets
  "OVMS does not run here", not "not on PATH".
- TUI files (optional, installed via the `[tui]` extra): `ovat/cli/tui.py`
  (launcher), `ovat/cli/shell.py` (subprocess exec layer + slash TEMPLATES),
  `ovat/cli/chat_screen.py` (in-process chat: streaming, history, sessions
  in `.ovat/sessions/`).

## HARD RULES: breaking these is how you nuke the codebase

1. **Branches**: everything lives on `main` now. The owner merged the TUI
   via PR #4 on 2026-07-04, so main IS the full toolkit (CLI + TUI). The
   old `week6-tui` branch is historical; do not develop on it. Never rebase
   or force-push anything. The TUI's separability is enforced by rule 3
   (the [tui] extra + isolation tests), not by branch topology anymore.
2. **Never push.** The owner pushes via his own tooling. Commit locally only.
3. **Isolation contract**: the CLI must fully work with NO TUI installed.
   `textual`/`pyfiglet` live in the `[tui]` extra only. No module-level
   `import textual` anywhere except `tui.py`/`chat_screen.py`.
   `tests/test_tui_isolation.py` enforces this: keep it passing.
4. **Single sources of truth**: colors → `ui.PALETTE` only; tool contracts →
   the co-located `SCHEMA` dicts only (LangChain arg models are derived);
   config validity → `workflow.py` (strict). Never create a parallel copy.
5. **Tests gate everything.** Run `python -m pytest -q` (use `.venv/bin/
   python`, Python 3.11; system 3.14 breaks OpenVINO). Both branches must
   end green: every fix ships with a test, every commit is one logical
   change with a conventional-commit message explaining WHY.
6. **Errors are for users**: tool/agent failures become readable strings the
   model (or human) can act on, never bare tracebacks in the CLI/TUI path.
7. Rollback tags exist: `v0.2.0-w5-6-midterm` (midterm state), `pre-tui-repair`
   (TUI before the big repair). Cut a new tag before anything risky.
8. The owner is a strong C++ dev but a Python beginner: explain changes in
   plain language (C++ analogies land well). Default is that HE writes the
   code with guidance; only write code for him when he explicitly asks.

## Environment variables

`OVAT_OVMS` (ovms binary/folder) · `OVAT_MODELS` (model discovery roots,
pathsep-separated) · `OVAT_VLM_MODEL` / `OVAT_WHISPER_MODEL` (tool model
dirs) · `OVAT_TEST_MODELS_DIR` / `OVAT_TEST_IMAGE` (integration tests) ·
`OVAT_TUI=1` (set by the TUI in children; makes bare `ovat` print a hint
instead of recursing).

## Platform truth (answer users honestly)

- **macOS**: dev + tests + `ovat chat` + TUI `/chat` + `index`/`doctor`/
  `init` all work (openvino_genai runs natively, CPU). OVMS does NOT exist
  on macOS: no `serve`, no `models`, no agentic `run`.
- **AI PC / Windows / Linux**: everything, including OVMS serving and
  tool-calling `run`. OVMS is usually NOT on PATH; the locator handles it.
- OVMS is x86-only; no Docker-OVMS on Apple Silicon (emulation crashes).

## Test suite conventions

`tests/conftest.py` has FakeLLMProvider/make_tool_call/reply. Markers:
`live` (needs running OVMS), `rag` (needs bge-small on disk); both
auto-skip. TUI tests use Textual's headless Pilot via `asyncio.run` (no
pytest-asyncio) and `pytest.importorskip("textual")`. Heavy seams for
mocking: `chat_screen._build_components`, `cli_main.build_agent`,
`model_server.subprocess.Popen`.

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
- Still open (proposal W7-W8): LlamaIndex engine, OpenAI Agents SDK engine,
  polished Document-Q&A sample agent, AI PC benchmarks (use --trace), CI
  (deliberately postponed; plan in the owner's Downloads).
