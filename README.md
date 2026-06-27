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
# 1. install (editable, from the repo root)
pip install -e .

# 2. scaffold a starter config you can edit
ovat init workflow.yml

# 3. on the AI PC (Windows/Linux), start OVMS serving your model
ovat serve workflow.yml

# 4. ask the agent something
ovat run workflow.yml --input "summarise my meeting notes"
```

No server handy? Prove the pipeline assembles without one:

```bash
ovat run workflow.yml --input "hi" --dry-run
# Built agent  model=Qwen3-8B-int4-ov  tools=['search_docs']  max_iterations=10
```

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
| `tools` | `name` / `type` | a built-in tool (`search_docs`, `transcribe`) |
| `agent` | `type` | `native` (the built-in loop) |
| | `max_iterations` | safety cap on tool-calling turns |
| | `system_prompt` | the agent's persona |

---

## Built-in tools

- **search_docs**: semantic search over local documents (vector retrieval).
- **transcribe**: speech-to-text on an audio file (OpenVINO Whisper).

Both are also standalone [MCP](https://modelcontextprotocol.io) servers, so any
MCP-aware agent can call them, not just OVAT.

---

## Status & limitations

Honest about where the abstraction holds and where it does not yet:

| Works today | Not yet |
| --- | --- |
| `ovat run/init/models/serve` CLI | LangChain front-end (`agent.type: react`) |
| YAML config + validation | External `mcp_stdio` tools (built-in only for now) |
| Native tool-calling loop + Session | macOS serving (OVMS is Windows/Linux only) |
| Built-in tools run in-process | Streaming responses |

OVMS runs on the Intel AI PC (Windows/Linux). On macOS you can develop and run
the unit tests, but not serve a model.

---

## Development

```bash
pip install -e ".[dev]"
pytest -m "not live"     # fast unit tests, no server needed (runs anywhere)
pytest -m live           # live tests against a running OVMS (AI PC only)
```

The codebase is layered: providers (swappable backends) → agent (loop, session,
factory) → config (YAML) → cli. Each new file carries comments explaining what
it does and why, so the next contributor can pick it up quickly.
