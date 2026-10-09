# OVAT design decisions

This is the log of design decisions behind OVAT: what was chosen, why, what
else was considered, and the measurement or failure that settled it.
[ARCHITECTURE.md](ARCHITECTURE.md) describes what the system is. This file
records how it got that way.

Each entry has four parts:

- **Decision**: what OVAT does.
- **Why**: the reason, in one or two sentences.
- **Alternatives considered**: what else was tried or weighed. Where the
  alternative was the earlier behaviour, it says so. Where none was recorded,
  it says that too.
- **Evidence**: the measurement, failure or source that decided it.

Entries are grouped by the part of the system they belong to, in the same
order as ARCHITECTURE.md.

## Contents

- [Configuration and the CLI](#configuration-and-the-cli)
- [Agent engines](#agent-engines)
- [Providers and RAG](#providers-and-rag)
- [Tools and MCP](#tools-and-mcp)
- [Serving OVMS](#serving-ovms)
- [Models and devices](#models-and-devices)
- [Telemetry](#telemetry)
- [Packaging and testing](#packaging-and-testing)

---

## Configuration and the CLI

### D1. Unknown config keys are errors

- **Decision:** `WorkflowConfig` derives from a `StrictModel` base with
  `extra="forbid"`.
- **Why:** a typo such as `max_iteration` for `max_iterations` should fail at
  once and by name, instead of quietly leaving the default in charge.
- **Alternatives considered:** pydantic's default, which ignores unknown keys.
- **Evidence:** the typo case above. The cost is accepted on purpose: every
  new field needs the schema, an example, a README row and a test.

### D2. One function loads the config

- **Decision:** `_load_config` in `cli/main.py` is the only caller of
  `load_workflow`.
- **Why:** it turns three different failures (file missing, YAML that does
  not parse, schema mismatch) into three different sentences, so every command
  reports them the same way.
- **Alternatives considered:** each command calling `load_workflow` itself,
  which is how it worked before.
- **Evidence:** before this, a bad workflow path printed a raw traceback from
  all five commands that read one. An empty or comments-only file gave
  `argument after ** must be a mapping`, because `yaml.safe_load` returns
  `None` for it; `yaml.safe_load(f) or {}` now turns that into "your config
  has no model section".

### D3. Model output is data, and is escaped before printing

- **Decision:** every value that reaches the console goes through `esc()`.
- **Why:** rich reads `[...]` as markup. Model answers, exception text and file
  paths are data, not markup.
- **Alternatives considered:** none recorded.
- **Evidence:** a model answer containing `[/INST]` crashed `ovat run` with a
  `MarkupError`.

---

## Agent engines

### D4. Every engine derives its tool arguments from the tool's `SCHEMA`

- **Decision:** `agent/arg_models.py` builds each framework's argument model
  from the `SCHEMA` dict that sits next to each tool.
- **Why:** one contract per tool means a tool cannot work on one engine and
  break on another.
- **Alternatives considered:** a hand-kept list of argument models per engine,
  which was tried once.
- **Evidence:** with the hand-kept list, a tool worked on the native loop and
  crashed on LangChain until someone remembered to update the second list.

### D5. One shared description of the OVMS connection

- **Decision:** `LLMBackend.from_config` (`providers/backend.py`) is the one
  place the four engines read the URL, model, timeout, temperature and token
  cap from.
- **Why:** settings that each engine copies drift apart.
- **Alternatives considered:** each engine reading the config itself, which
  is how it worked before.
- **Evidence:** for a whole release `temperature` was set in two engines and
  missing in the other two, so cross-engine timings compared greedy decoding
  against sampling rather than one framework against another.

### D6. The async engines refuse to run inside an existing event loop

- **Decision:** the LlamaIndex and OpenAI Agents engines are async. They call
  `asyncio.run` themselves, and raise a readable error if a loop is already
  running.
- **Why:** `asyncio.run` cannot be re-entered. Refusing gives a sentence;
  nesting would deadlock.
- **Alternatives considered:** nesting inside the caller's loop.
- **Evidence:** the TUI chat screen runs the agent on a worker thread, where
  no loop is running, and the engines work there.

### D7. The native loop decides on the reply's payload, not on `finish_reason`

- **Decision:** a reply that carries tool calls is a tool turn; a reply that
  carries none is the answer. `finish_reason` is not used to decide.
- **Why:** OVMS's [NPU demo](https://github.com/openvinotoolkit/model_server/blob/main/demos/llm_npu/README.md)
  documents that `finish_reason` is always `"stop"` on NPU, even for a decoded
  tool call.
- **Alternatives considered:** dispatching on `finish_reason == "tool_calls"`,
  which is what the hand-written loop in most examples does.
- **Evidence:** on OVMS 2026.2.1 on the LunarLake NPU (2026-08-12), the
  documented quirk did not occur: tool turns came back as `"tool_calls"`. So
  the defence follows the documentation but was not exercised on hardware.

### D8. Broken tool-call JSON goes back to the model

- **Decision:** if a tool call's arguments are not valid JSON, the loop does
  not run the tool. It returns the error as the tool result.
- **Why:** the model wrote the bad JSON, so the model is the one that can fix
  it on its next turn.
- **Alternatives considered:** parsing broken JSON as `{}`, which is what the
  loop did before.
- **Evidence:** with `{}`, the tool ran with missing arguments and the model
  never learned why.

### D9. Markdown fences around arguments are removed first

- **Decision:** `strip_code_fence` (in `ovat/text.py`) unwraps
  ```` ```json ```` before parsing. The native loop and the OpenAI Agents
  engine use it, because they are the two engines that parse arguments
  themselves.
- **Why:** small models wrap JSON in fences because that is how JSON looks in
  their training data. The packaging is wrong, the intent is not.
- **Alternatives considered:** rejecting fenced arguments.
- **Evidence:** rejecting them cost a whole round trip per call.

### D10. A misspelt tool name is matched, and the trace says so

- **Decision:** an unknown tool name is matched to the closest real one
  (`difflib`, 60% similarity). The trace records the tool that ran, and keeps
  the name the model asked for under `requested`.
- **Why:** a near miss such as `search` for `search_docs` is a clear intent.
- **Alternatives considered:** refusing unknown names outright (still the
  result when nothing is close enough).
- **Evidence:** before the trace fix, a matched call was logged under the name
  the model invented, which no tool has.

### D11. `Session` is thread-safe

- **Decision:** `Session` holds a lock. `save()` copies the history under the
  lock and writes the file outside it.
- **Why:** the TUI streams an answer and saves on a worker thread while the
  main thread can `/load` or `/clear`.
- **Alternatives considered:** none recorded.
- **Evidence:** `json.dump` over a list that another thread is appending to
  writes a truncated file, and that file is the user's saved conversation.
  Writing outside the lock keeps a slow disk from stalling the answer stream.

### D12. A run that cannot answer exits non-zero, on every engine

- **Decision:** `ovat run` exits with code 1 when the trace says `failed`,
  when a framework engine sets `last_failed`, or when the answer still holds
  raw tool-call markup.
- **Why:** the loop returns its failures as the answer text, because the model
  has to read them. Without an exit code a script cannot tell a failed run
  from a good one.
- **Alternatives considered:** exit 0 and print the error, which is how it
  worked before.
- **Evidence:** measured on the AI PC: four failed runs in a row, all exit 0.
  Reading only the native trace then let a capped LlamaIndex run exit 0, and
  OpenAI Agents returned raw markup and exited 0. Hence the three checks.

### D13. Undecoded tool-call markup is flagged, never parsed

- **Decision:** a reply that still contains `<tool_call>`, `<function=` or
  `<parameter=` is reported as `undecoded_tool_call` (or `truncated` if the
  reply was cut at `max_tokens`). OVAT does not try to turn the markup into a
  call itself.
- **Why:** guessing at broken output trades a loud failure for a silent wrong
  answer.
- **Alternatives considered:** parsing the markup client side.
- **Evidence:** `ovat bench` scores a row from these flags, not from the
  answer text. An earlier version tested the text and was defeated by a change
  in the same commit that turned an empty reply into an `Error:` string, which
  is not empty.

### D14. `model.max_tokens` defaults to 4096

- **Decision:** every engine sends a cap on the length of one reply, 4096
  tokens by default. Setting it to `null` restores the old behaviour.
- **Why:** a model that never emits a stop token otherwise generates until the
  client gives up (1200 s for `ovat run`, 600 s for a bench worker). Greedy
  decoding, which OVAT uses by default, makes that more likely.
- **Alternatives considered:** no cap, which was the behaviour until
  2026-08-12.
- **Evidence:** real answers here are 458 to 929 completion tokens. Without a
  cap, the KV cache climbed 0.62 to 3.6 GB in about nine minutes on one
  request, and one run reached 13.6 GB over an hour. The cache growth was the
  symptom, not the cause: an earlier reading blamed the cache, and the run
  that settled it had `native` fail first while `llamaindex` passed in 33 s.

### D15. `ovat chat` has no answer cap by default

- **Decision:** the local chat path streams until the model stops. Ctrl-C (or
  Esc in the TUI) ends it; `--max-tokens N` sets a cap.
- **Why:** the person is watching the answer and can stop it. The OVMS path
  keeps its cap because cancelling a request does not stop the generation
  inside the server.
- **Alternatives considered:** a fixed cap.
- **Evidence:** a fixed cap cut Qwen3.5 off in the middle of its reasoning.

---

## Providers and RAG

### D16. The factory returns the interface, not a concrete class

- **Decision:** `build_llm` returns `LLMProvider`.
- **Why:** a factory whose return type names one implementation can only ever
  produce that implementation.
- **Alternatives considered:** returning the OVMS class directly, which is how
  it was first written.
- **Evidence:** while the config had no `provider` field, `build_llm` could
  only ever return OVMS (see the comment on `ModelConfig.provider`).

### D17. Re-indexing a source replaces it

- **Decision:** `ovat index` deletes what a source had before adding it again.
- **Why:** indexing should be safe to repeat.
- **Alternatives considered:** appending, which is how it worked before.
- **Evidence:** three runs put the same chunk in three times, crowding out
  every other document.

### D18. A source is deleted from both tables

- **Decision:** deleting a source removes its rows from `chunks` and from the
  `vec0` virtual table.
- **Why:** `chunks` has a `source` column and `vec0` does not, so it is easy
  to clean only one.
- **Alternatives considered:** none recorded.
- **Evidence:** half-deleted sources leave orphan vectors. `retrieve` matches
  them and then skips them, so a request for `top_k=5` returns two results
  with no error.

### D19. The index remembers when it read each file

- **Decision:** `ovat index` writes `<db>.sources.json` with each file's path
  and modification time. `ovat run` and `ovat chat` warn about any file that
  changed or vanished.
- **Why:** a stale index answers just as confidently from text that is no
  longer there.
- **Alternatives considered:** none recorded. An index built before the
  manifest existed gets no warning.
- **Evidence:** the design reason above.

### D20. The vector query uses `k = ?`, not `LIMIT ?`

- **Decision:** the nearest-neighbour query is written in sqlite-vec's own
  `k = ?` form.
- **Why:** `LIMIT` only reaches a virtual table from SQLite 3.38, and Ubuntu
  22.04 ships 3.37.2.
- **Alternatives considered:** `LIMIT ?`, which is how it was first written.
- **Evidence:** on SQLite before 3.38, `ovat index` reported success, the query
  returned nothing, and the model answered with no citation. Every machine the
  code had been run on before had a newer SQLite.

---

## Tools and MCP

### D21. A tool's error has the same shape as its result

- **Decision:** a tool returns failures in the type its annotation declares.
  `search_docs` is `-> list[dict]`, so its errors are a list too.
- **Why:** FastMCP validates a tool's return value against the annotation.
- **Alternatives considered:** returning a plain error string, which is how it
  worked before.
- **Evidence:** the native loop hid the mismatch, because it calls `str()` on
  any result. Over MCP it was a crash: FastMCP rejected the string with "is not
  of type 'array'", so a locked database became a client-side exception.

### D22. MCP servers get a short environment, and `tools[].env` adds to it

- **Decision:** OVAT keeps the `mcp` SDK's short default environment (`HOME`,
  `LOGNAME`, `PATH`, `SHELL`, `TERM`, `USER`) and adds only the variables
  listed under `env:`. `${VAR}` is read from the user's shell.
- **Why:** a third-party server should not read every secret in the shell.
- **Alternatives considered:** passing the whole environment.
- **Evidence:** with the default alone, OVAT's own MCP servers never saw the
  variables they needed.

### D23. A misspelt file path is recovered, within limits, and announced

- **Decision:** `ovat/tools/fuzzy.py` uses the most similar existing file with
  the same extension (at least 70% alike). It searches only the named folder,
  or at most two levels below the working directory, and the tool result
  starts with "Note: there is no file at ... used the closest match ...".
- **Why:** small models misspell paths or drop a leading folder.
- **Alternatives considered:** an unbounded search, which an early version
  did; failing on any wrong path.
- **Evidence:** asked about `examples/audio-multimodal/sample.wav`, Qwen3.5
  sent `audio-multimodal/sample.wav` on every engine. The unbounded version
  walked a whole home folder and took 31.6 s.

### D24. An MCP-served `search_docs` builds its own retriever

- **Decision:** the MCP server takes `--config` and builds its own retriever
  from the workflow.
- **Why:** `type: mcp_stdio` starts a separate Python process, and objects do
  not cross a process boundary.
- **Alternatives considered:** passing the retriever object in (it cannot
  work across processes).
- **Evidence:** before the fix, the MCP-served tool was always in stub mode:
  104 characters of placeholder where the builtin path returned 1932 real ones
  from the same index.

### D25. One thread, one event loop and one manager coroutine per MCP server

- **Decision:** `tools/mcp_client.py` connects, serves and unwinds each server
  inside one long-lived coroutine. `close()` sets an event; the coroutine
  unwinds itself.
- **Why:** the `mcp` SDK is async (anyio) and OVAT's loop is not. anyio cancel
  scopes must be entered and exited by the same task.
- **Alternatives considered:** closing the connection from outside the task.
- **Evidence:** anyio's rule above; closing from another task breaks it.

---

## Serving OVMS

### D26. OVMS is pinned to 2026.4.1, chosen by measurement

- **Decision:** `ovms_installer.OVMS_VERSION` is `2026.4.1`.
- **Why:** it answered more tool-calling questions correctly and used far less
  memory than the previous pin.
- **Alternatives considered:** staying on 2026.2.1.
- **Evidence:** on the LunarLake AI PC, 15 questions that need a tool, through
  all four engines (2026-10-08, Windows only):

  | | OVMS 2026.2.1 | OVMS 2026.4.1 |
  | --- | --- | --- |
  | correct answers | 42 / 60 | **56 / 60** |
  | same answer on every repeat | no (up to 5 different answers in 5 runs) | yes |
  | peak memory of the GPU server | 17.3 GB | **4.9 GB** |

  14 answers went from wrong to right and none the other way (McNemar
  p = 0.0001). Linux has not been measured on 2026.4.1.

### D27. `setup` and `doctor` check the installed version

- **Decision:** both run `ovms --version` (under `ovms_env()`) and say when it
  differs from the pin. `ovat setup --force` replaces it.
- **Why:** a pin is useless if it never reaches machines that already have
  OVMS.
- **Alternatives considered:** checking only that a binary exists, which is
  how it worked before.
- **Evidence:** every machine with 2026.2.1 was told "already installed" and
  kept it. On Windows a bare `ovms --version` exits with no output, hence
  `ovms_env()`.

### D28. OVMS is installed by a subcommand, not inside the wheel

- **Decision:** `ovat setup` downloads OVMS on demand.
- **Why:** the archive is 126 to 185 MB, Linux needs one of three builds that
  cannot be chosen when the wheel is built, wheels have no post-install hook,
  and macOS has no build at all.
- **Alternatives considered:** bundling the binary in the wheel, which would
  charge every Mac user about 180 MB for a binary that cannot run.
- **Evidence:** the same shape is used by `playwright install` and
  `python -m spacy download`.

### D29. The installer flattens the archive and makes every file owner-writable

- **Decision:** the single `ovms/` folder in the archive is flattened into
  `~/.ovat/ovms`, and every member's owner-write bit is set before extraction.
- **Why:** unflattened, the binary lands one level below where the locator
  looks. OVMS ships files as mode `0o555`, and reopening such a file for write
  fails without `CAP_DAC_OVERRIDE`.
- **Alternatives considered:** extracting the archive as shipped.
- **Evidence:** install failed for every non-root Linux user, while passing in
  Docker, in CI and in a maintainer's container, all of which ran as root.

### D30. `ovms_env()` is a module-level function

- **Decision:** the environment for the OVMS child (`PATH`, `PYTHONHOME`,
  `LD_LIBRARY_PATH`) is built by `model_server.ovms_env()`, which can be tested
  without starting a server.
- **Why:** on Windows the `python_on` build loads `python3xx.dll` from
  `<ovms>/python`. On Linux the libraries are in `<root>/lib`, one level above
  the binary's own folder.
- **Alternatives considered:** building the environment inline in `start()`,
  which is how it was before.
- **Evidence:** while it was inline, the Linux branch had never run once,
  because every machine it was written on was Windows. Running the same binary
  twice settled it:

  | | exit | output |
  | --- | --- | --- |
  | bare | 127 | `libtbb.so.12: cannot open shared object file` |
  | under `ovms_env()` | 0 | `OpenVINO Model Server 2026.2.1` |

### D31. Readiness uses a stall budget, not a deadline

- **Decision:** `wait_until_ready` gives up only after 300 s with no progress.
  The clock resets whenever the log file or the model folder grows.
- **Why:** a first run downloads the model, and download time has no upper
  bound. Any fixed cap is either too small for a slow link or too large to
  notice a real hang.
- **Alternatives considered:** a fixed 120 s cap, which is how it worked
  before; a larger fixed cap.
- **Evidence:** the first download of Qwen3.5-4B alone took about 185 s. The
  same idea is behind curl's `--speed-time` and wget's `--read-timeout`.

### D32. OVMS writes its log to a file, never a pipe

- **Decision:** the child's output goes to `ovms.log`.
- **Why:** a pipe that nobody drains blocks the writer once the OS buffer
  (about 64 KB) fills, which would hang OVMS.
- **Alternatives considered:** a pipe read by a thread.
- **Evidence:** the pipe buffer limit above.

### D33. On Windows the server is detached from the console

- **Decision:** `ovat serve` starts OVMS with
  `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` on Windows, and `0` on POSIX.
- **Why:** `serve` hands the prompt back and leaves OVMS running. Without the
  flags, closing the window or pressing Ctrl-C in it stops the server too.
- **Alternatives considered:** none recorded.
- **Evidence:** the console behaviour above; `subprocess` requires `0` on
  POSIX.

### D34. `serve --stop` checks the process before signalling it

- **Decision:** `_pid_is_our_server` confirms the pid in `ovms.pid` is still
  an OVMS process before stopping it.
- **Why:** once a process is gone, the OS may reuse its number, so acting on
  the number alone could stop an unrelated program.
- **Alternatives considered:** checking only that the pid is alive.
- **Evidence:** pid reuse is standard OS behaviour.

### D35. `request_timeout` defaults to 1200 s

- **Decision:** one OVMS request may take up to 20 minutes before the client
  gives up.
- **Why:** an agent turn is up to `max_iterations` model calls, and CPU-only
  machines are slow on a cold start.
- **Alternatives considered:** 120 s, the earlier default.
- **Evidence:** on a CPU-only Ubuntu machine the first cold run took 1056 s,
  and the user saw "Request timed out." from a server that was working. The
  cost is that a hung server takes 20 minutes to time out, while
  `ovat doctor` finds it in seconds.

### D36. `ovms_cache_size_gb` is an integer

- **Decision:** the field is `int`.
- **Why:** OVMS declares `cache_size` as `uint64` and rejects `"1.0"`.
- **Alternatives considered:** `float`, which is how it was first typed.
- **Evidence:** with a float, OVMS refused to start for every value
  (`error parsing options: Argument '1.0' failed to parse`) and exited before
  opening its log, so the setting had never worked.

---

## Models and devices

### D37. "Unified" is its own kind of model

- **Decision:** `model_scout` reads `config.json` before judging the file
  layout. An exact `model_type` match (`qwen3_5`, `qwen3_5_text`, `qwen3_6`) marks the export
  as `unified`, which answers to both the `llm` and `vlm` filters and loads
  through `VLMPipeline`.
- **Why:** a unified export and a vision-only one look the same on disk.
- **Alternatives considered:** trusting the layout alone (the earlier
  behaviour); a prefix match on `model_type`, which would also accept a future
  vision-only `qwen3_5_vl`. Wrongly accepting a model gives a C++ traceback;
  wrongly rejecting one gives a readable sentence, so the match leans toward
  rejecting.
- **Evidence:** `LLMPipeline` builds successfully on a unified export (24.6 s)
  and then fails on the first `generate()` with "Port for tensor name
  input_ids was not found". A pipeline that builds is not proof it is the
  right one.

### D38. The default model is one unified export

- **Decision:** `Qwen3.5-4B-int4-ov` (3.5 GB) is the default. It does text,
  images and tool calls in one set of weights.
- **Why:** the RAG, ReAct and audio and vision examples share one download.
- **Alternatives considered:** a text LLM plus a separate vision model (about
  5 GB more).
- **Evidence:** measured on the AI PC, 4.3 GB steady and 6.5 GB peak, read
  from the OVMS process.

### D39. `tool_parser` is derived from the model name

- **Decision:** when `tool_parser` is omitted, OVAT picks it from the model
  name (`qwen3coder` for Qwen3.5, `hermes3` for Qwen3, `hermes3` for unknown
  families). An explicit value always wins.
- **Why:** the wrong parser fails silently: the agent answers fluently and
  never calls a tool.
- **Alternatives considered:** leaving it to OVMS (`auto`, or omitting the
  flag).
- **Evidence:** on OVMS 2026.2.1, `auto` selected no parser: the tool call came
  back as plain text with `finish_reason: "stop"` and zero tool calls. On
  2026.4.1, `auto` logged "Auto-detected tool_parser: qwen3coder" and scored
  54/60 against 56/60 for the derived value. OVAT still derives the name,
  because a named parser also works on an older OVMS installed by hand. The
  other families in the table (Qwen3.6, Qwen3-Coder, Phi-4-mini, Llama-3.2,
  gpt-oss, Devstral) come from OVMS's demos and were not measured here.

### D40. The device router suggests GPU for the agent model

- **Decision:** `DeviceManager` suggests GPU for the LLM when one is present,
  NPU for embeddings when one is present, and CPU for Whisper. A device
  written in `workflow.yml` always wins.
- **Why:** GPU works with every export. Serving an LLM on the NPU through OVMS
  needs a specific export and gives up batching and beam search.
- **Alternatives considered:** suggesting NPU for the LLM.
- **Evidence:** measured on LunarLake with OVMS 2026.2.1 (2026-08-12):
  `OpenVINO/Qwen3-8B-int4-cw-ov` compiled for NPU in 36 s and ran a
  tool-calling agent (`tool_calls: 1`, 2 turns, 63.8 s). The stock
  `Qwen3.5-0.8B-int4-ov` failed with `0x78000004 - [NPU_VCL]`. The compiler's
  own message was `Found 8 duplicated names`, a graph-naming complaint, so
  group quantisation as the cause is not verified. What is verified is that the
  channel-wise export compiles and the stock one does not.

---

## Telemetry

### D41. Absent is not zero

- **Decision:** an unknown number (token counts, peak memory) is `null` and
  shows as a dash.
- **Why:** `0` reads as "used no tokens", and a benchmark built on it is
  wrong without anyone noticing.
- **Alternatives considered:** reporting `0`, which is what an early trace
  did.
- **Evidence:** the telemetry audit of 2026-07-29 found unknown tokens
  reported as `0`.

### D42. Peak memory is sampled on a thread during the run

- **Decision:** `_PeakMemory` samples resident memory on a background thread
  for the whole run.
- **Why:** one reading after the run misses the peak, because Python has
  already freed the large allocations.
- **Alternatives considered:** one reading at the end, which is how it worked
  before.
- **Evidence:** the same audit found "peak RSS" was not a peak.

### D43. `ovat bench` runs each engine in its own process

- **Decision:** each engine gets a fresh subprocess.
- **Why:** resident memory is a whole-process number, so engines sharing a
  process inherit each other's allocations.
- **Alternatives considered:** running all engines in one process.
- **Evidence:** against live OVMS, the native engine read 465.8 MB running
  first and 1155.6 MB running last, against 466 MB alone. Reversing the order
  reversed the conclusion.

### D44. An unavailable source says why

- **Decision:** a source that cannot read its hardware returns nothing and a
  reason, for example "Intel Unified Telemetry does not run on macOS".
- **Why:** a missing sensor and an idle one look the same as zeros in a graph.
- **Alternatives considered:** showing zeros.
- **Evidence:** the design reason above.

### D45. The TUI telemetry status line refreshes every tick

- **Decision:** the Live tab's status line names each source as live, silent
  or not available, and is redrawn every tick.
- **Why:** a source's state can change after start-up.
- **Alternatives considered:** a separate tab filled once at start, which is
  how it worked before.
- **Evidence:** on the AI PC the old tab said "intel live" while Intel UT was
  producing nothing.

### D46. Windows NPU utilisation comes from the `GPU Engine` counter

- **Decision:** `_WindowsNPUCounter` reads `\GPU Engine(*)\Utilization Percentage`
  for the Intel AI Boost adapter, through PDH.
- **Why:** Windows exposes the NPU as a compute adapter under that counter.
- **Alternatives considered:** `Get-Counter -ListSet *NPU*`, which matches
  "User Input Delay" (the "npu" inside "I-npu-t"); the GPU's own
  `engtype_neural` engine.
- **Evidence:** two controls, both measured: 93.5% under an NPU load, and 0.0%
  while OVMS generated on the GPU, when the GPU's `engtype_neural` read 100.0%.

### D47. Per-request tracing comes from plano, not from code in OVAT

- **Decision:** OVAT has no OpenTelemetry dependency and no OTLP exporter.
  Per-request OpenTelemetry spans come from the optional plano gateway, which
  runs outside OVAT.
- **Why:** plano already emits spans for every request it forwards, so OVAT
  gets them from configuration rather than code.
- **Alternatives considered:** an OTLP exporter inside OVAT. None exists yet;
  it would be another telemetry sink.
- **Evidence:** [`examples/plano/`](../examples/plano/) runs it, including the
  id bridge OVMS needs.

---

## Packaging and testing

### D48. Optional dependencies stay optional, and CI proves it

- **Decision:** `textual` and `pyfiglet` live only in the `[tui]` extra, and
  the framework engines only in their own extras. A CI job installs the base
  package and checks that importing the CLI pulls in none of them.
- **Why:** a base install must run without any of them.
- **Alternatives considered:** relying on the test suite alone.
- **Evidence:** the dev install has every extra, so the suite cannot see a
  stray top-level import. One such import crashes a base install on start-up.

### D49. Shared text helpers live in a neutral module

- **Decision:** `ovat/text.py` holds the helpers for reading model output
  (reasoning blocks, undecoded tool calls, code fences).
- **Why:** both the agent core and the CLI/TUI need them, and neither may
  import the other.
- **Alternatives considered:** keeping them in the UI, which is where they
  began.
- **Evidence:** the helpers started in `chat_screen.py`, moved to `cli/ui.py`
  when the plain CLI needed them, and moved here when `loop.py` did too.

### D50. Tests are isolated from the machine they run on

- **Decision:** disk-scanning tests use `monkeypatch.chdir` and a fake `HOME`,
  and `live` and `rag` tests skip when their server or model is missing.
- **Why:** a test that depends on the machine describes the machine, not the
  code.
- **Alternatives considered:** none recorded.
- **Evidence:** three real cases. The stall-budget tests called the real health
  URL, so they failed wherever `ovat serve` happened to be running. Two
  telemetry tests asserted a source was unavailable, which is true on macOS and
  false on an AI PC. One config test resolved a filename against the working
  tree and passed while the documented command was still broken.

### D51. Every fix ships with a test that fails without it

- **Decision:** a fix is not done until its test fails with the fix backed
  out.
- **Why:** a test that passes against broken code gives false confidence.
- **Alternatives considered:** none recorded.
- **Evidence:** it has happened in this project more than once.
