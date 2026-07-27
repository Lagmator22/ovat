# OVAT: OpenVINO Agentic Toolkit

Turn agent boilerplate into **one YAML file + one command**.

OVAT runs a tool-calling AI agent on an Intel AI PC, backed by
[OpenVINO Model Server (OVMS)](https://docs.openvino.ai/2025/model-server/ovms_what_is_openvino_model_server.html).
You describe the model, the tools, and the agent in a small `workflow.yml`,
then run it:

```bash
ovat run workflow.yml --input "What do my notes say about Q3?"
```

> GSoC 2026 · Intel / OpenVINO · Project #18. This is a work in progress; see
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
| `tools` | `name` / `type` | `builtin` (`search_docs`, `transcribe`) or `mcp_stdio` |
| | `command` | (for `mcp_stdio`) how to launch the MCP server |
| `agent` | `type` | `native` (built-in loop) or `react` (LangChain) |
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
| Run traces: `ovat run --trace` (tokens, latency, RSS) | |
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
