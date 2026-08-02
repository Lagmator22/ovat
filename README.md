# OVAT: OpenVINO Agentic Toolkit

Turn agent boilerplate into **one YAML file + one command**.

OVAT runs a tool-calling AI agent on an Intel AI PC, backed by
[OpenVINO Model Server (OVMS)](https://docs.openvino.ai/2025/model-server/ovms_what_is_openvino_model_server.html).
You describe the model, the tools, and the agent in a small `workflow.yml`,
then run it:

```bash
ovat run workflow.yml --input "What do my notes say about Q3?"
```

> OpenVINO | Project #18 | This is a work in progress; see
> [Status](#status--limitations) for what works today.

---

## Why OVAT? (the abstraction, in one screen)

A "simple" tool-calling agent against OVMS is really ~50 lines of boilerplate:
build the OpenAI client, hand-write each tool's JSON schema, run the
call → check `finish_reason` → dispatch the tool → append the result → loop,
and manage the message history yourself. Every new agent copy-pastes it and
diverges.

**Without OVAT**, every project re-writes this:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v3", api_key="x")
tools = [ { "type": "function", "function": { "name": "search_docs",
            "parameters": { ... } } } ]            # hand-written schema
messages = [{"role": "user", "content": question}]
while True:                                        # the loop, by hand
    r = client.chat.completions.create(model="...", messages=messages, tools=tools)
    choice = r.choices[0]
    if choice.finish_reason != "tool_calls":
        print(choice.message.content); break
    for call in choice.message.tool_calls:         # dispatch, by hand
        result = run_my_tool(call.function.name, call.function.arguments)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    # ...plus max-iteration guard, error handling, history management...
```

**With OVAT**, you write this `workflow.yml`:

```yaml
model:
  name: Qwen3.5-4B-int4-ov
  device: GPU
  ovms_url: http://localhost:8000/v3
  tool_parser: auto
tools:
  - name: search_docs
    type: builtin
agent:
  type: native
  max_iterations: 10
```

…and run `ovat run workflow.yml --input "..."`. The loop, schemas, history, and
error handling are the toolkit's job now.

**The payoff is config, not code.** Moving from a 16 GB GPU box to an 8 GB
CPU laptop is a three-line edit. Compare [`workflow.yml`](examples/workflow.yml)
(Standard / GPU) with [`minimal.yml`](examples/minimal.yml) (Minimal / CPU).
The agent never changes; only the YAML does.

---

## Prerequisites

Check these **before** installing. Skipping the driver step is the most
common reason a GPU or NPU device never shows up in `ovat doctor`.

| | Requirement | Notes |
| --- | --- | --- |
| **Python** | 3.10 – 3.14 | `python --version`. Wheels exist for all of these. |
| **Git** | any | Optional (only needed for developer installation). |
| **OS for the full agent** | Windows 11, Ubuntu 22.04/24.04, RHEL 9 | OVMS is x86-64 only. |
| **OS for development** | + macOS (Apple Silicon or Intel) | Everything except serving. See [Platform support](#platform-support). |
| **RAM** | 8 GB minimum, 16 GB comfortable | The default 4B model wants ~5 GB; a ~2 GB tier is documented below. Above 4B with long prompts, prefer 16 GB. |
| **Disk** | ~8 GB free | Default model 3.5 GB, OVMS 126 MB zipped (more unpacked), Python deps ~2 GB, plus your index. |

### Intel GPU / NPU drivers

The CPU works with no driver work at all. For **GPU or NPU** on an Intel AI PC,
update the driver first — an out-of-date driver typically shows up as the
device simply being missing from `ovat doctor`, not as an error.

| Device | Windows | Linux |
| --- | --- | --- |
| **GPU** (Arc / Iris Xe) | [Intel Arc & Iris Xe driver](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html) | [GPU configuration guide](https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-gpu.html) |
| **NPU** (Core Ultra) | [Intel NPU driver](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html) | [NPU configuration guide](https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-npu.html) |

On **Windows** also install the
[Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/VC_redist.x64.exe),
which OVMS needs to start at all.

Verify afterwards — the devices OpenVINO can actually see are listed by:

```bash
ovat doctor workflow.yml     # look at the "OpenVINO devices" row
```

> **NPU cannot do tool calling.** Use `device: CPU` or `device: GPU` for
> agent workflows. NPU is fine for embeddings and for plain chat.

---

## Install

### 1. OVAT itself

Install OVAT and the interactive TUI directly from PyPI. (We recommend using a virtual environment):

```bash
python -m venv .venv
# Activate it:
#   Mac/Linux:   source .venv/bin/activate
#   Windows:     .\.venv\Scripts\Activate.ps1

pip install "ovat[tui]"
```

**For Developers (Optional)**

If you want to edit the code or contribute, install from a clone instead:

```bash
git clone https://github.com/Lagmator22/ovat.git
cd ovat

python -m venv .venv
# Activate it (see above)

python -m pip install -e ".[langchain,llamaindex,openai-agents,tui]"
```

Extras, so you can install only what you need:

| Extra | Gives you |
| --- | --- |
| *(none)* | the native engine, all built-in tools, MCP, RAG, telemetry |
| `langchain` / `llamaindex` / `openai-agents` | the matching `agent.type` |
| `tui` | the full-screen launcher (`ovat` with no arguments) |
| `convert` | `optimum-cli`, to convert HuggingFace models to OpenVINO IR |
| `dev` | every framework above, plus pytest |

### 2. OpenVINO Model Server (Windows / Linux)

`ovat serve`, and every tool-calling `run`, needs OVMS. It is a single
archive — no installer, and it does **not** go on `PATH`; OVAT finds it for
you (see [`ovms_locator.py`](ovat/core/ovms_locator.py)).

> **Download the `python_on` build.** The `python_off` (C++ only) package
> **cannot do tool calling** — Intel's own docs state that its limited chat
> template support means "using tools is not possible". Since tool calling is
> the entire point of OVAT, the wrong archive produces an agent that answers
> normally but silently never calls a tool.

**Windows 11**

```bat
curl -L https://github.com/openvinotoolkit/model_server/releases/download/v2026.2.1/ovms_windows_2026.2.1_python_on.zip -o ovms.zip
tar -xf ovms.zip
.\ovms\setupvars.bat
```

**Ubuntu 24.04** (use `ubuntu22` or `redhat` in the filename for those)

```bash
wget https://github.com/openvinotoolkit/model_server/releases/download/v2026.2.1/ovms_ubuntu24_2026.2.1_python_on.tar.gz
tar -xzvf ovms_ubuntu24_2026.2.1_python_on.tar.gz
sudo apt update && sudo apt install -y libxml2 curl
export LD_LIBRARY_PATH=${PWD}/ovms/lib
export PATH=$PATH:${PWD}/ovms/bin
export PYTHONPATH=${PWD}/ovms/lib/python
```

> Do **not** `pip install openvino`, `openvino-tokenizers` or `openvino-genai`
> into the interpreter that OVMS's `PYTHONPATH` points at — the `python_on`
> package ships its own copies and mixing them breaks both. OVAT's own venv is
> separate, so a normal OVAT install is unaffected.

Not on `PATH`? That is expected. Point OVAT at the folder once:

```bash
export OVAT_OVMS=/path/to/ovms          # or set model.ovms_binary in the YAML
```

macOS has no native OVMS build. See [Platform support](#platform-support).

### 3. A model

Two ways to get one; both land in the same place.

**a. Let `ovat serve` fetch it** (needs OVMS; uses `ovms --pull`):

```bash
ovat serve workflow.yml        # downloads on first run, then serves
```

**b. Download the IR directly** (works anywhere, including macOS):

```bash
hf download OpenVINO/Qwen3.5-4B-int4-ov \
    --local-dir models/OpenVINO/Qwen3.5-4B-int4-ov
```

These are already-converted OpenVINO IR models — no conversion step, no
`optimum-cli`. Pick the tier that matches your machine:

| Model | Download | RAM (est.) | Use it when |
| --- | --- | --- | --- |
| [`OpenVINO/Qwen3.5-4B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3.5-4B-int4-ov) | **3.5 GB** | ~4.5–5.5 GB | **Default.** Text + vision + tools in one model. |
| [`OpenVINO/Qwen3.5-0.8B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3.5-0.8B-int4-ov) | **0.9 GB** | ~1.5–2 GB | 8 GB laptop, or you want a fast first run. |
| [`OpenVINO/Qwen3-8B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3-8B-int4-ov) | 4.9 GB | ~6–7 GB | Strongest text answers; no vision. |
| [`OpenVINO/whisper-base-int8-ov`](https://huggingface.co/OpenVINO/whisper-base-int8-ov) | 0.08 GB | small | The `transcribe` tool. |

> **How the RAM column is derived**, since a wrong number here wastes a
> download: it is *weights + KV cache + 15–20% runtime overhead*, the standard
> INT4 estimate, applied to the weights actually on disk — not to the download
> size. Those differ, and for these models it matters. A Qwen3.5 export is
> **five** models, not one: `language_model` is 2.32 GB of the 4B's 3.24 GB,
> with the rest in the text and vision embedding models that load alongside it.
>
> The KV cache grows with context, so long prompts cost more than the table
> shows. Intel notes that models **above** 4B with prompts over 1024 tokens can
> want more than 16 GB
> ([release notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html)).
> Treat these as planning figures; measure on your own hardware with
> `ovat run --trace`, which reports sampled peak RSS.

The Qwen3.5 models are **unified**: one export does text generation, image
understanding *and* tool calling, so the RAG, ReAct and multimodal examples
all share a single download. They are also thinking models — they reason
before answering, and OVAT folds that reasoning away in the TUI.

The embedding model for RAG is the one thing that still needs converting,
because no pre-built OpenVINO IR of it is published:

```bash
pip install "ovat[convert]"
optimum-cli export openvino --model BAAI/bge-small-en-v1.5 \
    --task feature-extraction models/bge-small-en-v1.5
```

---

## Quickstart

With the three steps above done:

```bash
# 1. scaffold a starter config you can edit
ovat init workflow.yml

# 2. check the machine and that config together
#    (Python, deps, devices, OVMS, the YAML itself)
ovat doctor workflow.yml

# 3. prove the pipeline assembles -- no server, no model needed
ovat run workflow.yml --input "hi" --dry-run
# Built agent  model=Qwen3.5-4B-int4-ov  tools=['search_docs', 'transcribe']  max_iterations=10

# 4. index your documents so search_docs can find them (optional)
ovat index ./my-notes workflow.yml

# 5. Windows/Linux: start OVMS. Returns once it is READY and leaves it
#    running in the background (pid in ovms.pid, logs in ovms.log)
ovat serve workflow.yml

# 6. ask the agent something
ovat run workflow.yml --input "summarise my meeting notes"

# 7. done for the day? stop the background OVMS cleanly
ovat serve workflow.yml --stop
```

On **macOS**, steps 5 and 6 have no OVMS. Use the local path instead — note it
uses [`examples/rag/workflow.yml`](examples/rag/workflow.yml), not the starter
file: `ovat chat` answers *from your index*, so it needs a config with a `rag:`
section, and the starter ships that block commented out.

```bash
# the embedder, once (there is no pre-built OpenVINO IR of bge-small)
pip install "ovat[convert]"
optimum-cli export openvino --model BAAI/bge-small-en-v1.5 \
    --task feature-extraction models/bge-small-en-v1.5

hf download OpenVINO/Qwen3.5-0.8B-int4-ov --local-dir models/Qwen3.5-0.8B-int4-ov
ovat index ./examples/rag/docs examples/rag/workflow.yml
ovat chat examples/rag/workflow.yml --input "What is OVAT's memory budget?"
# → "Under 8 GB", and the source file it came from
```

### Three worked examples

| Use case | Folder | What it shows |
| --- | --- | --- |
| **RAG** | [`examples/rag/`](examples/rag/) | Ask questions about your own documents, with citations. |
| **ReAct** | [`examples/react/`](examples/react/) | The same agent through LangChain, and all four engines compared. |
| **Audio + vision** | [`examples/audio-multimodal/`](examples/audio-multimodal/) | Transcribe a `.wav`, describe an image, in one agent. |

---

## Interactive launcher (TUI)

The TUI is an **optional** front-end and is already included in the full
install above. To add it to a base project installation, install the `tui`
extra, then run `ovat` with no arguments:

```bash
python -m pip install -e ".[tui]"
ovat
```

It opens in the terminal's alternate screen, like a separate window, so it never
smears into your scrollback. Type `/` to see the commands, Tab to complete:

```
OpenVINO Agentic Toolkit
  /chat [config] [model]    chat with your indexed docs (native, streaming)
  /doctor [config]          check Python, deps, devices, OVMS, a config
  /init [path]              write a starter workflow.yml you can edit
  /validate <config>        load and validate a workflow file (via doctor)
  /index <folder> <config>  index a folder of docs for search_docs
  /telemetry                live CPU, memory and Intel hardware numbers
  /run  /serve  /help  /models  /clear  /exit
```

Anything else you type runs as a real shell command in the venv (`pytest`,
`git status`, `ls`), with streamed output; Esc cancels a running command.

`/chat` is the flagship: a native chat screen that loads the model **once**
and keeps it warm, streams the answer token by token, remembers the
conversation across turns, and autosaves it under `.ovat/sessions/` (`/save
<name>` and `/load <name>` for named conversations). Your config and model
path are remembered in `.ovat/chat_prefs.json`, so after the first time a
bare `/chat` just works.

**Isolation contract:** the plain CLI never needs the TUI. Installing without
the `tui` extra pulls in no TUI dependencies at all, every subcommand works
identically, and a bare `ovat` prints a pointer instead of a launcher. The
TUI can be adopted, or removed, without touching the toolkit.

---

## The workflow file

| Section | Field | Meaning |
| --- | --- | --- |
| `model` | `name` | model name OVMS serves |
| | `device` | `CPU`, `GPU`, or `NPU` |
| | `ovms_url` | where OVMS listens |
| | `tool_parser` | how tool calls are decoded. `auto` lets OVMS read the chat template and choose — prefer it, since naming one overrides OVMS's own detection and the right answer differs per family (`hermes3` for Qwen3, `qwen3coder` for Qwen3.5) |
| | `source_model` | (for `ovat serve`) HF id to download/serve |
| | `model_repository_path` | (for `ovat serve`) folder where models live |
| `tools` | `name` / `type` | `builtin` (`search_docs`, `transcribe`, `describe_image`) or `mcp_stdio` |
| | `command` | (for `mcp_stdio`) how to launch the MCP server |
| `agent` | `type` | `native`, `react`, `llamaindex`, or `openai-agents` ([all four](#four-engines-one-yaml-word)) |
| | `max_iterations` | safety cap on tool-calling turns |
| | `system_prompt` | the agent's persona |
| `rag` | `embeddings` | the embedder: `provider` (`genai`/`ovms`), `model`, `device`, `dim` |
| | `retriever` | the vector store: `provider` (`sqlite-vec`), `db_path` |
| | `chunk` | `size` and `overlap` in characters |

The `rag` section is optional. Leave it out and `search_docs` runs in stub mode;
add it and the tool returns real chunks with citations.

---

## Four engines, one YAML word

`agent.type` chooses how the loop runs, and nothing else in your config changes:

- `native`: OVAT's own tool-calling loop (`loop.py`). Zero extra dependencies.
- `react`: the same job through **LangChain** (`create_agent` + `ChatOpenAI`
  pointed at OVMS). Install it with `pip install 'ovat[langchain]'`.
- `llamaindex`: the same job through **LlamaIndex** (`FunctionAgent` +
  `OpenAILike` pointed at OVMS). Install it with `pip install 'ovat[llamaindex]'`.
- `openai-agents`: the same job through the **OpenAI Agents SDK** compatibility
  model. Install it with `pip install 'ovat[openai-agents]'`.

All four expose the same configuration and tools; swapping is a one-word edit.

For a single run, override `agent.type` without touching the file. The YAML is
**never rewritten** — the next run goes back to whatever it says:

```bash
ovat run workflow.yml -i "..." --llamaindex        # bare flag
ovat run workflow.yml -i "..." --engine llamaindex # or name it explicitly
ovat bench workflow.yml -i "..." --engines native,react   # or compare them
```

Each engine answers to its library name too, because that is what people
reach for first:

| Flag | Also | Engine |
| --- | --- | --- |
| `--native` | | `native` |
| `--react` | `--langchain` | `react` |
| `--llamaindex` | | `llamaindex` |
| `--openai-agents` | `--openai-sdk` | `openai-agents` |

Naming two different engines is an error rather than a coin flip, so a trace
can never report a framework that did not produce it.

---

## RAG: search your own documents

`search_docs` is real retrieval, swappable by config. The embedder and the
vector store are chosen by **string**, honouring the provider abstractions, so
moving from a local embedder to a server-side one is a YAML edit, not a code
change.

```bash
# Export an OpenVINO embedding model once (~130 MB). optimum-cli lives in the
# `convert` extra; there is no pre-built OpenVINO IR of bge-small to download.
pip install "ovat[convert]"
optimum-cli export openvino --model BAAI/bge-small-en-v1.5 \
    --task feature-extraction models/bge-small-en-v1.5

ovat index ./my-notes workflow.yml      # chunk + embed + store, with sources
ovat run workflow.yml --input "what did the Q3 review conclude?"
```

A full worked version of this, with sample documents, is in
[`examples/rag/`](examples/rag/).

Each result carries its source file, so the agent can cite where an answer came
from.

### Chat locally, no server (macOS / dev)

OVMS does not run on macOS, but `openvino_genai` does. `ovat chat` answers from
your index with a **local** OpenVINO model, so you can test real RAG without a
server (no tool-calling; it always retrieves then answers):

Both commands need a config carrying a `rag:` section — `ovat chat` answers
from an index, so it exits early without one. `examples/rag/workflow.yml` has
it; the file `ovat init` writes ships that block commented out.

```bash
hf download OpenVINO/Qwen3.5-0.8B-int4-ov --local-dir models/Qwen3.5-0.8B-int4-ov
ovat index ./my-notes examples/rag/workflow.yml
ovat chat examples/rag/workflow.yml --model-path models/Qwen3.5-0.8B-int4-ov \
    --input "what did the Q3 review conclude?"
# → an answer grounded in your notes, with the source files listed
```

Omit `--model-path` and OVAT finds a usable model itself, scanning
`OVAT_MODELS`, `./models` and `~/models`.

The full agentic path (the model deciding to call tools) still uses OVMS on the
AI PC; `ovat chat` is the local retrieval-augmented fallback.

---

## Tools: built-in and MCP

- **search_docs**: semantic search over your local documents with source
  citations (vector retrieval via the `rag` config).
- **transcribe**: speech-to-text on an audio file (OpenVINO Whisper).
- **describe_image**: caption or answer questions about an image
  (OpenVINO Qwen2-VL). Point `OVAT_VLM_MODEL` at the exported model folder.

Both are also standalone [MCP](https://modelcontextprotocol.io) servers, so any
MCP-aware agent can call them, not just OVAT.

And the door swings both ways: OVAT speaks MCP as a **client**. Declare a tool
with `type: mcp_stdio` and OVAT launches the server, discovers every tool it
advertises, and hands them to the agent exactly like built-ins:

```yaml
tools:
  - name: search_docs          # our own tool, over the wire this time
    type: mcp_stdio
    command: ["python", "-m", "ovat.tools.search_docs"]
  - name: anything_else        # ANY third-party MCP server plugs in the same way
    type: mcp_stdio
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/docs"]
```

---

## Document Q&A and four-engine benchmark

[`examples/document-qa.yml`](examples/document-qa.yml) is the local Document
Q&A sample. On an AI PC with OVMS running, index the files once and then send
the same question through every engine:

```bash
ovat index ./docs examples/document-qa.yml
ovat serve examples/document-qa.yml
ovat bench examples/document-qa.yml -i "what does the readme say about NPU?" \
  --out report.json
```

The terminal table compares build time, answer time, peak memory, and any
available token/tool counts. `report.json` preserves each engine's answer and
the full failure text when one engine cannot run.

---

## Telemetry: what the run actually cost

Every run can report what it spent. Sources (where numbers come from) and
sinks (where they go) are separate contracts, so any source feeds any sink:

| Source | Reports | Works on |
| --- | --- | --- |
| `agent` | tokens per turn, latency, tool-call traces | any engine with a native loop |
| `system` | CPU per core, RAM, thread count | macOS, Windows, Linux |
| `process` | this process's resident memory | macOS, Windows, Linux |
| `intel` | GPU/NPU utilisation and power | Windows / Linux (Intel UT) |

```bash
ovat run workflow.yml --input "..." --trace trace.json      # one run's trace
ovat run workflow.yml --input "..." --telemetry live.jsonl  # sampled alongside
ovat telemetry                                              # live table, no run
ovat telemetry --out metrics.jsonl                          # ...and to disk
ovat                                                        # then /telemetry
```

Output is JSON Lines (one object per line): a run killed half-way still leaves
a readable file, and `tail -f` works while it runs.

Two rules the numbers follow, because a measurement that lies is worse than no
measurement:

- **Unknown stays unknown.** If the server never reports token counts, the
  field is `null` and renders as a dash. A `0` would read as "used no tokens".
- **An unavailable source says why.** On macOS the Intel row reads *"Intel
  Unified Telemetry does not run on macOS"* rather than showing zeros, because
  a missing sensor and an idle one look identical in a graph.

The Intel source needs [Intel Unified Telemetry](https://github.com/intel/ut).
Unzip the release and set `OVAT_UT` to that folder (or drop it in `~/ut`, which
OVAT finds on its own). Known limit, measured on the AI PC: UT's continuous
mode writes binary traces that need `bin2perfetto` to decode, so it reports as
running but does not yet stream numbers.

---

## OpenTelemetry via the plano gateway (optional)

[plano](https://github.com/katanemo/plano) (formerly archgw) is an AI proxy
that sits between OVAT and OVMS and turns every request into an OpenTelemetry
span — latency, TTFT, token counts — with **no OTEL dependency added to OVAT**.

[`examples/plano/`](examples/plano/) has the full working setup. Three things
had to be solved, and each answer is in the config's comments:

| Problem | Answer |
| --- | --- |
| plano calls `/v1`, OVMS serves `/v3` | No prefix setting exists: plano parses `base_url` and lifts the path out itself. Put `/v3` in the URL. |
| plano refuses to start | The model name needs a `provider/` prefix (plano splits on `/`). Hence `ovms/Qwen3.5-4B-int4-ov`. |
| plano rejects OVMS's reply | Its WASM filter requires a top-level `"id"`, which OVMS omits. [`ovms_id_bridge.py`](examples/plano/ovms_id_bridge.py) injects one. |

plano ships Linux and macOS binaries only — there is **no Windows build**, so
on the AI PC it runs under WSL2 or `planoai up --docker` while OVMS runs
natively on the Windows host for the Arc GPU.

### Which host address to use

`base_url` in [`plano-config.yaml`](examples/plano/plano-config.yaml) must name
the machine the **bridge** runs on, and that address depends on where plano
itself is running. Find yours in the table, run the command, paste the result.

| plano runs… | OVMS + bridge run… | `base_url` host |
| --- | --- | --- |
| same Linux box | same Linux box | `127.0.0.1` — no lookup needed |
| WSL2 | Windows host | the WSL gateway IP (below) |
| WSL2 (mirrored networking) | Windows host | `127.0.0.1` |
| Docker (`--docker`) | the host | `host.docker.internal` |
| another machine | the AI PC | the AI PC's LAN IP |

```bash
# Plain Linux, everything on one box — nothing to look up:
echo 127.0.0.1

# WSL2 reaching the Windows host: the default gateway is the host.
ip route | grep default | awk '{print $3}'          # e.g. 172.22.64.1

# Same thing, straight into the config (edit-in-place, keeps a .bak):
sed -i.bak "s|base_url: http://[^:]*:|base_url: http://$(ip route | grep default | awk '{print $3}'):|" \
    examples/plano/plano-config.yaml

# Docker on Linux: if host.docker.internal does not resolve, start plano with
#   --add-host=host.docker.internal:host-gateway
# or use the docker0 bridge address:
ip -4 addr show docker0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}'    # e.g. 172.17.0.1
```

```cmd
:: The AI PC's own LAN address, for reaching it from another machine
ipconfig | findstr /C:"IPv4 Address"
```

Two things that bite:

- **The WSL gateway IP changes when WSL restarts.** If plano suddenly returns
  connection errors after a reboot, re-run the `ip route` command and update
  `base_url`. On Windows 11 22H2+ you can avoid this entirely by enabling
  mirrored networking (`networkingMode=mirrored` in `.wslconfig`), after which
  `127.0.0.1` reaches the host directly.
- **Windows Firewall blocks the bridge port by default.** Open it once, from
  an *elevated* Command Prompt on the Windows host:

  ```cmd
  netsh advfirewall firewall add rule name="OVAT bridge 8001" ^
      dir=in action=allow protocol=TCP localport=8001
  ```

Check the whole path before starting plano — this should print JSON, not hang:

```bash
curl -s http://<host>:8001/v3/models
```

### Run it

```bash
ovat serve examples/plano/workflow.yml            # OVMS on :8000
python examples/plano/ovms_id_bridge.py           # bridge on :8001
planoai up examples/plano/plano-config.yaml       # plano on :12000
ovat run examples/plano/workflow.yml --input "hello"
planoai obs                                       # live OTEL dashboard
```

See [`examples/plano/README.md`](examples/plano/README.md) for the cross-OS
networking details.

---

## Status & limitations

Honest about where the abstraction holds and where it does not yet:

| Works today | Not yet |
| --- | --- |
| `ovat run/chat/init/index/serve/models/doctor` CLI | macOS *serving* (OVMS is Windows/Linux only) |
| Full-screen TUI with a native streaming chat screen | Streaming from OVMS (local GenAI streams already) |
| YAML config + strict validation (typos are errors) | Re-ranking / hybrid search |
| Native, LangChain (`react`), LlamaIndex, and OpenAI Agents SDK engines | Approximate vector backends (usearch/hnsw) |
| External MCP tools (`type: mcp_stdio`, any server) | |
| Real RAG in `search_docs` (vectors + citations) | |
| Local RAG chat + model auto-detection (no OVMS) | |
| OVMS lifecycle: `ovat serve` + `--stop` (pidfile, no PATH edits) | |
| Run traces: `ovat run --trace` (tokens, latency, RSS) | Intel UT streaming (writes binary traces; needs `bin2perfetto`) |
| Telemetry sources + JSONL export, CLI and TUI page | OVMS Docker integration tests, stress tests |
| OpenTelemetry through the plano gateway (config, not code) | |
| `ovat doctor` platform-aware diagnostics | CI |
| Unified multimodal models (one export: text + vision + tools) | |
| Published PyPI package |

### Platform support

| | macOS | Windows 11 | Linux (Ubuntu/RHEL) |
| --- | --- | --- | --- |
| `init` `doctor` `index` `validate` | ✅ | ✅ | ✅ |
| `chat` (local model, no server) + TUI | ✅ | ✅ | ✅ |
| `serve` / `models` (OVMS) | ❌ | ✅ | ✅ |
| `run` / `bench` (tool-calling agent) | ❌ | ✅ | ✅ |
| GPU / NPU acceleration | ❌ (CPU only) | ✅ | ✅ |

OVMS is x86-64 only and has **no macOS build**. On Apple Silicon the official
Docker image does run under Rosetta emulation — verified booting and serving a
real model — which is fine for small models but slow for an 8B-class LLM. For
day-to-day macOS work, `ovat chat` with a local `openvino_genai` model is the
supported path and needs no server at all.

---

## Documentation

| Document | What is in it |
| --- | --- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The layered design, all four engines, both serving paths, the telemetry layer, and the plano gateway. |
| [`examples/rag/`](examples/rag/) | Retrieval-augmented Q&A over your own documents. |
| [`examples/react/`](examples/react/) | The ReAct engine, and the same question across all four engines. |
| [`examples/audio-multimodal/`](examples/audio-multimodal/) | Whisper transcription plus image understanding. |
| [`examples/plano/`](examples/plano/) | OpenTelemetry through the plano AI gateway. |
| [`AGENTS.md`](AGENTS.md) | Contributor notes: hard rules, landmines, and why things are the way they are. |

---

## Development

```bash
pip install -e ".[dev]"   # dev pulls in every optional agent framework
pytest -m "not live"      # fast unit tests, no server needed (runs anywhere)
pytest -m live            # live tests against a running OVMS (AI PC only)
pytest -m "not rag"       # skip the real-embedding-model test if it is not exported
```

Test markers: `live` needs a running OVMS server; `rag` needs the bge-small
model on disk. Both auto-skip when their dependency is absent, so a fresh clone
runs green out of the box.

The codebase is layered: providers (swappable backends) → agent (loop, session,
factory) → config (YAML) → cli. Each new file carries comments explaining what
it does and why, so the next contributor can pick it up quickly.
