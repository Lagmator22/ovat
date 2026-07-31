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
  name: Qwen3-8B-int4-ov
  device: GPU
  ovms_url: http://localhost:8000/v3
  tool_parser: hermes3
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

## Quickstart

```bash
# 1a. project virtual environment (recommended; from the repo root).
python -m venv .venv
# Activate it: PowerShell: .\.venv\Scripts\Activate.ps1
#              bash/zsh:    source .venv/bin/activate
python -m pip install -e ".[langchain,llamaindex,openai-agents,tui]"

# 1b. equivalent isolated global installation (choose this instead of 1a).
pipx install ".[langchain,llamaindex,openai-agents,tui]"

# 2. check your machine is ready (Python, deps, devices, OVMS, a config)
ovat doctor workflow.yml

# 3. scaffold a starter config you can edit
ovat init workflow.yml

# 4. index your documents so search_docs can find them
ovat index ./my-notes workflow.yml

# 5. on the AI PC (Windows/Linux), start OVMS serving your model.
#    serve returns once OVMS is READY and leaves it running in the
#    background (pid recorded in ovms.pid, logs in ovms.log)
ovat serve workflow.yml

# 6. ask the agent something
ovat run workflow.yml --input "summarise my meeting notes"

# 7. done for the day? stop the background OVMS cleanly
ovat serve workflow.yml --stop
```

No server handy? Prove the pipeline assembles without one:

```bash
ovat run workflow.yml --input "hi" --dry-run
# Built agent  model=Qwen3-8B-int4-ov  tools=['search_docs', 'transcribe']  max_iterations=10
```

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

**Isolation contract:** the plain CLI never needs the TUI. `pip install ovat`
without the extra installs no TUI dependencies at all, every subcommand works
identically, and a bare `ovat` prints a pointer instead of a launcher. The
TUI can be adopted, or removed, without touching the toolkit.

---

## The workflow file

| Section | Field | Meaning |
| --- | --- | --- |
| `model` | `name` | model name OVMS serves |
| | `device` | `CPU`, `GPU`, or `NPU` |
| | `ovms_url` | where OVMS listens |
| | `tool_parser` | how tool calls are decoded (`hermes3` for Qwen3) |
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

---

## RAG: search your own documents

`search_docs` is real retrieval, swappable by config. The embedder and the
vector store are chosen by **string**, honouring the provider abstractions, so
moving from a local embedder to a server-side one is a YAML edit, not a code
change.

```bash
# export an OpenVINO embedding model once (any optimum-cli export works)
optimum-cli export openvino --model BAAI/bge-small-en-v1.5 \
    --task feature-extraction models/bge-small-en-v1.5

ovat index ./my-notes workflow.yml      # chunk + embed + store, with sources
ovat run workflow.yml --input "what did the Q3 review conclude?"
```

Each result carries its source file, so the agent can cite where an answer came
from.

### Chat locally, no server (macOS / dev)

OVMS does not run on macOS, but `openvino_genai` does. `ovat chat` answers from
your index with a **local** OpenVINO model, so you can test real RAG without a
server (no tool-calling; it always retrieves then answers):

```bash
ovat index ./my-notes workflow.yml
ovat chat workflow.yml --model-path models/Llama-3.2-3B-Instruct-INT4 \
    --input "what did the Q3 review conclude?"
# → an answer grounded in your notes, with the source files listed
```

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
| plano refuses to start | The model name needs a `provider/` prefix (plano splits on `/`). Hence `ovms/Qwen3-8B-int4-ov`. |
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
| `ovat doctor` platform-aware diagnostics | |

OVMS runs on the Intel AI PC (Windows/Linux). On macOS you can develop and run
the unit tests, but not serve a model.

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
