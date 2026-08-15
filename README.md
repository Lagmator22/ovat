# OVAT: OpenVINO Agentic Toolkit

**Build a tool-calling AI agent on an Intel AI PC from one YAML file and one command.**

[![PyPI](https://img.shields.io/pypi/v/ovat?v=1.0.1)](https://pypi.org/project/ovat/)
[![Python](https://img.shields.io/pypi/pyversions/ovat?v=1.0.1)](https://pypi.org/project/ovat/)
[![License](https://img.shields.io/pypi/l/ovat?v=1.0.1)](LICENSE)

```bash
pip install ovat
ovat setup                                    # install the model server, once
ovat init workflow.yml
ovat serve workflow.yml                       # start it
ovat run workflow.yml -i "what do my notes say about Q3?"
```

Everything runs locally on your own hardware, through
[OpenVINO Model Server](https://docs.openvino.ai/2026/model-server/ovms_what_is_openvino_model_server.html).
No API keys, no cloud, nothing leaving the machine.

> **GSoC 2026 · OpenVINO Project #18.** Full agent on Windows and Linux with an
> Intel CPU/GPU/NPU; macOS is supported for development. See
> [Platform support](#platform-support).

---

## Contents

- [Why OVAT](#why-ovat)
- [Prerequisites](#prerequisites) · [Install](#install) · [Get a model](#get-a-model)
- [Quickstart](#quickstart)
- [Examples](#examples): RAG, ReAct, audio + vision
- [The workflow file](#the-workflow-file)
- [Four engines, one config](#four-engines-one-config)
- [Tools](#tools) · [Telemetry](#telemetry)
- [Platform support](#platform-support) · [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

---

## Why OVAT

A tool-calling agent against OVMS is ~50 lines of boilerplate every project
rewrites: build the client, hand-write each tool's JSON schema, call, check
`finish_reason`, dispatch the tool, append the result, loop, guard the iteration
count, manage history.

OVAT makes that a config file:

```yaml
model:
  name: Qwen3.5-4B-int4-ov
  device: GPU
  tool_parser: qwen3coder
tools:
  - name: search_docs
    type: builtin
agent:
  type: native
  max_iterations: 10
```

The loop, schemas, history and error handling are the toolkit's job now. Moving
from a 16 GB GPU box to an 8 GB CPU laptop is a three-line edit: compare
[`examples/workflow.yml`](examples/workflow.yml) with
[`examples/minimal.yml`](examples/minimal.yml).

---

## Prerequisites

| | Needs | Notes |
| --- | --- | --- |
| **Python** | 3.10, 3.14 | `python3 --version` (see below on Linux) |
| **OS (full agent)** | Windows 11, Ubuntu 22.04/24.04, RHEL 9 | OVMS is x86-64 only |
| **OS (development)** | + macOS | everything except serving |
| **RAM** | 8 GB min, 16 GB comfortable | the default model wants ~5 GB |
| **Disk** | ~8 GB, or ~15 GB with RAG | RAG pulls torch via the `convert` extra |

### Linux: install Python and OVMS's system libraries first

A minimal Linux image (a container, a fresh VM, a server install) ships with
**no Python at all**, and OVMS needs `libxml2`. Verified in a clean
`ubuntu:24.04` container, where `python3`, `pip` and `venv` are all absent:

```bash
# Ubuntu 22.04 / 24.04  (the releases Intel builds OVMS for)
sudo apt update && sudo apt install -y python3 python3-venv python3-pip libxml2 curl

# RHEL 9 / Rocky / Alma
sudo dnf install -y python3 python3-pip libxml2 curl
```

> **Ubuntu 26.04 and newer are not supported yet.** There is no OVMS build for
> them, the package is named `libxml2-16` rather than `libxml2`, and the
> ubuntu24 archive cannot start there because it needs that release's system
> libraries. `ovat setup` warns before downloading. Use 22.04 or 24.04.

On Linux the interpreter is **`python3`**, not `python`, and `python3-venv` is
a separate package from `python3` on Debian and Ubuntu. Then:

```bash
python3 -m venv .venv && source .venv/bin/activate
```

Everything after this point is the same on every platform.

**Using GPU or NPU?** Update the driver first, an old driver usually shows up as
the device simply being absent from `ovat doctor`, not as an error.

| Device | Windows | Linux |
| --- | --- | --- |
| GPU (Arc / Iris Xe) | [driver](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html) | [setup guide](https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-gpu.html) |
| NPU (Core Ultra) | [driver](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html) | [setup guide](https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-npu.html) |

On Windows also install the
[Visual C++ Redistributable](https://aka.ms/vs/17/release/VC_redist.x64.exe),
which OVMS needs in order to start at all.

> **Agents default to `GPU`, and `NPU` needs a specific export.** No
> accelerator executes tools: the device runs the model, the agent loop parses
> the tool call out of the generated text and runs the Python function itself.
> Tool calling is a property of the model and the parser, not the hardware --
> OVMS serves tool-calling LLMs on NPU and documents how.
>
> What the NPU requires is a **channel-wise symmetric INT4** export
> (`--sym --ratio 1.0 --group-size -1`). The stock `-int4-ov` models are group
> quantised and fail to compile on it with `[NPU_VCL] Compilation failed
> (0x78000004)` -- measured here on both Qwen3.5-4B and Qwen3.5-0.8B. Use the
> [`-int4-cw-ov` family](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu)
> instead:
>
> ```bash
> ovms --pull --source_model OpenVINO/Qwen3-8B-int4-cw-ov \
>   --model_repository_path models --target_device NPU \
>   --task text_generation --tool_parser hermes3 --max_prompt_len 2000
> ```
>
> NPU also gives up request batching, beam search and `log_probs`, always
> reports `finish_reason: "stop"` (OVAT handles that: it dispatches on the tool
> call itself, not on the label), and caps the prompt at 1024 tokens unless
> `--max_prompt_len` raises it. `GPU` stays the default because it is correct
> for any export. Embeddings route to NPU automatically.

---

## Install

```bash
pip install ovat
```

Extras, so you install only what you need:

| Extra | Gives you |
| --- | --- |
| *(none)* | native engine, all built-in tools, MCP, RAG, telemetry |
| `langchain` · `llamaindex` · `openai-agents` | the matching `agent.type` |
| `tui` | the full-screen launcher (`ovat` with no arguments) |
| `convert` | `optimum-cli`, to convert HuggingFace models to OpenVINO IR |
| `dev` | every framework above, plus pytest |

```bash
pip install "ovat[langchain,llamaindex,openai-agents,tui]"
```

### OpenVINO Model Server (Windows / Linux)

`ovat serve`, and every tool-calling `run`, needs OVMS. One command installs it:

```bash
ovat setup
```

That picks the right archive for your OS (and, on Linux, your distro), checks
its SHA-256, and unpacks it into `~/.ovat/ovms` - a folder OVAT already
searches. **Nothing is added to `PATH` and no environment variable is needed.**
Run it once per machine; it is safe to run again.

On macOS it prints why there is nothing to install and what to use instead —
Intel ships Windows and Linux x86-64 builds only.

If you skip this step, `ovat serve` notices OVMS is missing and offers to do it
for you. It never downloads unattended: with no terminal attached (CI, a pipe)
it stops and tells you to run `ovat setup`.

<details>
<summary>Install it by hand instead (air-gapped machines, or a build you already have)</summary>

> ⚠️ **Take the `python_on` build.** The `python_off` (C++ only) package
> **cannot do tool calling**. Intel's own docs state that its limited
> chat-template support means "using tools is not possible". The wrong archive
> gives you an agent that answers normally and silently never calls a tool.

**Windows 11** (run this from the folder you want OVMS in):

```bat
curl -L https://github.com/openvinotoolkit/model_server/releases/download/v2026.2.1/ovms_windows_2026.2.1_python_on.zip -o ovms.zip
tar -xf ovms.zip
```

**Ubuntu 24.04** (swap `ubuntu22` or `redhat` as needed):

```bash
wget https://github.com/openvinotoolkit/model_server/releases/download/v2026.2.1/ovms_ubuntu24_2026.2.1_python_on.tar.gz
tar -xzvf ovms_ubuntu24_2026.2.1_python_on.tar.gz
sudo apt update && sudo apt install -y libxml2 curl
```

`ovat serve` sets the library paths for you. If you launch `ovms` **yourself**
on Linux, it needs them exported first, or it cannot load its own `.so` files:

```bash
export LD_LIBRARY_PATH=${PWD}/ovms/lib
export PYTHONPATH=${PWD}/ovms/lib/python
```

OVAT searches `./ovms`, `~/.ovat/ovms`, `~/ovms_windows`, `~/ovms`, `C:\ovms`,
and `PATH`. If yours lives elsewhere:

```bash
export OVAT_OVMS=/path/to/ovms        # or set model.ovms_binary in the YAML
```

</details>

macOS has no OVMS build. See [Platform support](#platform-support).

---

## Get a model

`ovat serve` downloads one for you on the first run. To fetch it yourself
(the only option on macOS):

```bash
hf download OpenVINO/Qwen3.5-4B-int4-ov --local-dir models/OpenVINO/Qwen3.5-4B-int4-ov
```

These are pre-converted OpenVINO IR, no conversion needed.

| Model | Download | RAM | Use it when |
| --- | --- | --- | --- |
| [`Qwen3.5-4B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3.5-4B-int4-ov) | **3.5 GB** | 4.3 GB steady, **6.5 GB peak** | **Default.** Text + vision + tools in one model |
| [`Qwen3.5-0.8B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3.5-0.8B-int4-ov) | **0.9 GB** | ~2 GB | 8 GB machine, or a fast first run |
| [`Qwen3-8B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3-8B-int4-ov) | 4.9 GB | ~6-7 GB | strongest text answers; no vision |
| [`whisper-base-int8-ov`](https://huggingface.co/OpenVINO/whisper-base-int8-ov) | 0.08 GB | small | the `transcribe` tool |


**Which of these are measured.** The Qwen3.5-4B row is measured on an Intel AI PC, read from the OVMS process (the model's memory lives there, not in OVAT). Download sizes come from the model cards. Every figure written with `~` is an ESTIMATE from parameter count and precision, not a measurement -- if you need a number you can rely on for capacity planning, measure it on your own machine with `ovat run --telemetry`.

> **These are steady-state figures for a short answer.** A long generation grows
> the KV cache as it goes, and that is not included above. Measured on an AI PC
> during a runaway 18-minute request, the KV cache grew 3.6 → 5.8 GB and
> `ovms.exe` reached **10.6 GB** resident. Cap it with `agent.max_iterations`
> and the model's own answer length if you are near your RAM limit; the steady
> figures are what to plan for, not a ceiling.

The 4B figures are **measured on an Intel AI PC**. The peak sits 2.2 GB above
steady state, and the load spike is what decides whether a model fits, so 8 GB
machines should prefer the 0.8B tier.

Qwen3.5 is a **unified** model: text, images and tool calling in one export, so
all three examples share a single download.

RAG also needs an embedder, the one thing that has to be converted:

```bash
pip install "ovat[convert]"
optimum-cli export openvino --model BAAI/bge-small-en-v1.5 \
    --task feature-extraction models/bge-small-en-v1.5
```

---

## Quickstart

```bash
ovat setup                                    # 1. install OVMS (once per machine)
ovat init workflow.yml                        # 2. write a starter config
ovat doctor workflow.yml                      # 3. check machine + config
ovat run workflow.yml -i "hi" --dry-run       # 4. no server, no model needed
ovat serve workflow.yml                       # 5. start OVMS
ovat run workflow.yml -i "summarise my notes" # 6. ask something
ovat serve workflow.yml --stop                # 7. shut it down
```

**Step 5 takes a while the first time**, it downloads the model. That is
expected: `serve` shows elapsed time and only gives up after five minutes of *no
progress at all*, so a slow link is fine.

Skipping step 1 is fine too - `ovat serve` will offer to install OVMS when it
finds none.

On **macOS** there is no OVMS. Use the local path instead. `ovat chat` answers
from an index, so it needs a config with a `rag:` section, which the example
below provides:

```bash
git clone https://github.com/Lagmator22/ovat.git && cd ovat   # for the examples
hf download OpenVINO/Qwen3.5-0.8B-int4-ov --local-dir models/Qwen3.5-0.8B-int4-ov

pip install "ovat[convert]"                                   # for the embedder
optimum-cli export openvino --model BAAI/bge-small-en-v1.5 \
    --task feature-extraction models/bge-small-en-v1.5

ovat index ./examples/rag/docs examples/rag/workflow.yml
ovat chat examples/rag/workflow.yml -i "What is OVAT's memory budget?"
```

---

## Examples

| Use case | Folder | Shows |
| --- | --- | --- |
| **RAG** | [`examples/rag/`](examples/rag/) | Answers from your own documents, with citations |
| **ReAct** | [`examples/react/`](examples/react/) | The same agent through LangChain; all four engines compared |
| **Audio + vision** | [`examples/audio-multimodal/`](examples/audio-multimodal/) | Transcribe a `.wav`, describe an image |
| **OpenTelemetry** | [`examples/plano/`](examples/plano/) | Traces through the plano AI gateway |

Each folder has its own README with the exact commands.

> The examples are **not** included in the pip package, since they are project
> files rather than library code. Clone the repo to run them:
> `git clone https://github.com/Lagmator22/ovat.git && cd ovat`. Everything else
> in this README works from `pip install ovat` alone.

---

## The workflow file

| Section | Field | Meaning |
| --- | --- | --- |
| `model` | `name` | the model OVMS serves |
| | `device` | `CPU`, `GPU`, or `NPU` |
| | `ovms_url` | where OVMS listens (default `http://localhost:8000/v3`) |
| | `ovms_port` | internal port for `ovat serve` when fronted by a proxy (default 8000) |
| | `tool_parser` | how tool calls are decoded, **`qwen3coder`** for Qwen3.5, `hermes3` for Qwen3. Derived from the model name if omitted |
| | `source_model` | HF id that `ovat serve` downloads |
| | `request_timeout` | per-request cap, in seconds |
| | `ovms_max_prompt_len` | longest prompt OVMS accepts. **Needed on NPU**, where OVMS caps it at 1024 and one agent turn exceeds that; defaults to 4096 on NPU and is left to OVMS elsewhere |
| | `max_tokens` | ceiling on one reply (default `4096`). `null` restores the old unbounded behaviour, where a model that never emits a stop token generates until the client gives up |
| `tools` | `name` / `type` | `builtin` (`search_docs`, `transcribe`, `describe_image`) or `mcp_stdio` |
| | `model` | where that tool's weights live. `transcribe` and `describe_image` only |
| | `device` | `CPU`, `GPU` or `NPU` for that tool; omit to let the device router choose |
| `agent` | `type` | `native`, `react`, `llamaindex`, `openai-agents` |
| | `max_iterations` | safety cap on tool-calling turns |
| | `system_prompt` | the agent's persona |
| `model_search_paths` | *(list)* | extra folders to scan for local model exports, before `./models` and `~/models`. Replaces `OVAT_MODELS` |
| `rag` | `embeddings` / `retriever` / `chunk` | vector search for `search_docs` |
| | `retriever.provider` | `sqlite-vec` (default, persists to `db_path`) or `memory` |
| | `retriever.db_path` | where `sqlite-vec` keeps the index; ignored by `memory` |

`rag` is optional. Without it `search_docs` answers in a documented stub mode, so
every quickstart command works on a fresh install with nothing downloaded.

The table above is the tour. Every field, with its type, default and
constraints, is in
[`docs/workflow_yaml_reference.md`](docs/workflow_yaml_reference.md).

The `memory` retriever keeps no file, so nothing survives the process: `ovat
index` followed by a separate `ovat run` retrieves nothing. Use it when indexing
and querying happen in one run, or when you want a demo that leaves no
`ovat_index.db` behind. `sqlite-vec` is the default because that separation is
the normal way RAG gets used here.

---

## Four engines, one config

`agent.type` picks the loop; nothing else in your config changes.

- **`native`**. OVAT's own loop. Zero extra dependencies, and the only engine
  that records per-turn token counts.
- **`react`**. LangChain (`create_agent` + `ChatOpenAI` pointed at OVMS)
- **`llamaindex`**. LlamaIndex `FunctionAgent` + `OpenAILike`
- **`openai-agents`**. OpenAI Agents SDK compatibility model

Override for a single run without editing the file:

```bash
ovat run workflow.yml -i "..." --llamaindex
ovat run workflow.yml -i "..." --react              # or --langchain
ovat run workflow.yml -i "..." --openai-sdk         # or --openai-agents
ovat bench workflow.yml -i "..." --out report.json  # all four, side by side
```

`bench` prints build time, answer time, peak memory and token counts. A failing
engine is a **row, not a crash**, and an engine that answered nothing scores
**not ok** rather than a misleading green.

---

## Tools

- **`search_docs`**, semantic search over your documents, returning source paths
- **`transcribe`**, speech-to-text from any OpenVINO export; set the tool's `model:`
- **`describe_image`**, caption or answer questions about an image; set the tool's `model:`

All three are also standalone [MCP](https://modelcontextprotocol.io) servers, so
any MCP-aware agent can call them. And OVAT is an MCP **client**, point it at
any server:

```yaml
tools:
  - name: search_docs
    type: mcp_stdio
    command: ["python", "-m", "ovat.tools.search_docs", "--config", "workflow.yml"]
  - name: anything_else
    type: mcp_stdio
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/docs"]
```

> Pass `--config` when serving `search_docs` over MCP. That server runs in its
> own process and builds its own retriever; without a config it stays in stub
> mode.

---

## Telemetry

```bash
ovat run workflow.yml -i "..." --trace trace.json      # one run's trace
ovat run workflow.yml -i "..." --telemetry live.jsonl  # sampled alongside
ovat telemetry                                         # live table, no run
```

Two rules the numbers follow:

- **Unknown stays unknown.** If the server reports no token counts the field is
  `null`, not `0`, a zero would read as "used no tokens".
- **An unavailable source says why**, instead of reporting zeros.

`--trace` peak RSS measures the **OVAT process**. With OVMS serving, the model
lives in `ovms.exe`, so measure that instead (Task Manager, or
`Get-Process ovms`). `--trace` *is* the right number for `ovat chat`, where the
model runs in-process.

---

## Platform support

| | macOS | Windows 11 | Linux |
| --- | --- | --- | --- |
| `init` `doctor` `index` `setup` `telemetry` | ✅ | ✅ | ✅ |
| `chat` (local model, no server) + TUI | ✅ | ✅ | ✅ |
| `serve` `models` `run` `bench` | ❌ | ✅ | ✅ |
| GPU / NPU | ❌ CPU only | ✅ verified | ⚠️ untested |

✅ means run on that platform. Linux GPU/NPU is marked untested because the
Linux verification was done in WSL2, which exposes no `/dev/dri` -- so the CPU
path is proven there and the accelerator path is not. It is expected to work
with current drivers; it has not been demonstrated here.

OVMS is x86-64 with no macOS build. For day-to-day macOS work, `ovat chat` runs a
local `openvino_genai` model and needs no server at all.

---

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Agent answers fluently but **never calls a tool** | Wrong `tool_parser`. Qwen3.5 needs `qwen3coder`. OVAT warns rather than hiding this |
| `OVMS exited without becoming ready`, empty log | The `python_off` build, or a missing VC++ Redistributable |
| `ovat doctor` finds no OVMS | Set `OVAT_OVMS` to the unpacked folder |
| No GPU/NPU listed in `doctor` | Driver out of date. See [Prerequisites](#prerequisites) |
| `search_docs` returns `[stub]` | No `rag:` section, or no `--config` on the MCP command |
| `serve` looks stuck | The first run downloads the model. Watch `ovms.log` |

`ovat doctor <config>` is the fastest diagnosis for anything else.

---

## Documentation

| | |
| --- | --- |
| [`docs/workflow_yaml_reference.md`](docs/workflow_yaml_reference.md) | Every `workflow.yml` field: type, default, constraints and purpose. Generated from the schema |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The nine layers, all four engines, both serving paths, and the design decisions behind them |
| [`docs/BLOG.md`](docs/BLOG.md) | A walkthrough: what OVAT is and how to use it |
| [`examples/`](examples/) | Four runnable use cases |
| [`AGENTS.md`](AGENTS.md) | Contributor notes: hard rules and landmines |

---

## Development

```bash
git clone https://github.com/Lagmator22/ovat.git && cd ovat
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate.bat
pip install -e ".[dev]"
pytest -q                    # ~550 tests, no server needed
pytest -m live               # against a running OVMS (AI PC only)
```

`live` needs OVMS; `rag` needs bge-small on disk. Both auto-skip, so a fresh
clone runs green.

Licensed under [Apache 2.0](LICENSE).
