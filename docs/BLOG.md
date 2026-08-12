# Build a local AI agent on your Intel laptop with one YAML file

*OVAT, the OpenVINO Agentic Toolkit, turns the boilerplate of a tool-calling
agent into a config file, and runs the whole thing on your own hardware.*

---

## Contents

1. [The problem: most of an agent is plumbing](#1-the-problem-most-of-an-agent-is-plumbing)
2. [What OVAT gives you](#2-what-ovat-gives-you)
3. [Why OpenVINO and OVMS](#3-why-openvino-and-ovms)
4. [Install and first run](#4-install-and-first-run)
5. [Example 1: Ask your own documents (RAG)](#5-example-1-ask-your-own-documents-rag)
6. [Example 2: Swap the agent framework in one word](#6-example-2-swap-the-agent-framework-in-one-word)
7. [Example 3: Listen and look (audio + vision)](#7-example-3-listen-and-look-audio--vision)
8. [Choosing a model for your machine](#8-choosing-a-model-for-your-machine)
9. [Where the time and memory actually go](#9-where-the-time-and-memory-actually-go)
10. [Extending OVAT with your own tools](#10-extending-ovat-with-your-own-tools)
11. [What it is not](#11-what-it-is-not)
12. [Try it](#12-try-it)

---

## 1. The problem: most of an agent is plumbing

An "AI agent" is a simple idea: a model that can call your functions. Ask it
about a file, and instead of guessing, it calls `search_docs` and reads the
answer back to you.

**The thing worth changing is not how much code that takes — it is that it
takes code at all.** Building an agent should be a matter of stating what you
want: this model, on this device, with these tools and this system prompt.
Instead it means picking an orchestration framework, learning its API, wiring a
model server to it, and rewriting the same loop each time you change your mind
about any of those.

OVAT makes those choices **configuration**. The framework is one word in a YAML
file, and swapping it changes nothing else — not your tools, not your prompt,
not a line of Python. That is the difference: you spend your attention on the
tools, the prompt and the model, and none of it on the machinery underneath.

Here is the machinery you would otherwise own.

The idea is simple. The code is not. Here is the smallest honest version of a
tool-calling agent against a local model server:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v3", api_key="not-needed")

tools = [{"type": "function", "function": {
    "name": "search_docs",
    "description": "Search the user's local documents",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string"}},
                   "required": ["query"]}}}]           # hand-written schema

messages = [{"role": "user", "content": question}]
while True:                                            # the loop, by hand
    r = client.chat.completions.create(
        model="Qwen3.5-4B-int4-ov", messages=messages,
        tools=tools, tool_choice="auto")
    choice = r.choices[0]
    if choice.finish_reason != "tool_calls":
        print(choice.message.content)
        break
    messages.append(choice.message)
    for call in choice.message.tool_calls:             # dispatch, by hand
        args = json.loads(call.function.arguments)     # may be malformed
        result = run_my_tool(call.function.name, args)
        messages.append({"role": "tool",
                         "tool_call_id": call.id,
                         "content": str(result)})
```

That is already fifty lines with the interesting parts removed. Still missing: an
iteration cap so a confused model cannot loop forever, error handling so a
failing tool does not kill the run, session history if you want a second turn,
device selection, and starting the model server in the first place.

None of that is your product. Every project writes it, every copy diverges, and
none of it is where the value is.

---

## 2. What OVAT gives you

OVAT is a pip-installable Python library and CLI. You describe the agent; it
builds it.

```yaml
# workflow.yml
model:
  name: Qwen3.5-4B-int4-ov
  device: GPU
  tool_parser: qwen3coder
  source_model: OpenVINO/Qwen3.5-4B-int4-ov

tools:
  - name: search_docs
    type: builtin

agent:
  type: native
  max_iterations: 10
  system_prompt: >-
    You answer questions about the user's own documents. Always call
    search_docs first, and cite the source path you used.
```

```bash
ovat serve workflow.yml                       # start the model server
ovat run workflow.yml -i "what did the Q3 review conclude?"
```

That is the whole thing. The loop, the schemas, the history, the retries and the
error messages are the toolkit's job. Ten commands cover the rest:

| Command | Does |
| --- | --- |
| `ovat init` | write a starter `workflow.yml` |
| `ovat doctor` | check Python, devices, drivers, OVMS, and your config |
| `ovat serve` / `--stop` | start and stop the model server |
| `ovat run` | ask the agent a question |
| `ovat chat` | local retrieve-then-answer, no server needed |
| `ovat index` | embed a folder of documents |
| `ovat bench` | run one question through every engine, side by side |
| `ovat models` | list and pull models |
| `ovat telemetry` | live CPU / RAM / Intel hardware numbers |
| `ovat` | a full-screen terminal UI |

---

## 3. Why OpenVINO and OVMS

OVAT does not do inference. That is
[OpenVINO Model Server](https://docs.openvino.ai/2026/model-server/ovms_what_is_openvino_model_server.html)'s
job, and OVMS is already very good at the hard parts:

- **An OpenAI-compatible `/v3` endpoint**, so any client that speaks OpenAI works
- **Tool-call decoding** via `--tool_parser`, in C++, not Python string-munging
- **Continuous batching and prefix caching**
- **Real device targeting**, the same model on CPU, integrated GPU, or NPU

What OVMS deliberately does not do is decide *for you*: which model, which
device, which parser, which tools, when to start and stop. That gap is where a
toolkit belongs, the same way `kubectl` wraps the Kubernetes API rather than
replacing it.

OpenVINO matters for the other half: **low-bit weights (INT4, INT8) that actually fit**.
A 4B-parameter model in INT4 is a 3.5 GB download and runs on an integrated GPU.
That is the difference between "you need a datacentre" and "you need a laptop".

OVAT auto-detects what you have via `openvino.Core().get_available_devices()` and
routes accordingly:

| Model type | Goes to | Why |
| --- | --- | --- |
| LLM | GPU | dynamic shapes, KV cache, tool calling |
| Embeddings | NPU if present | small, static shape, low power |
| Whisper | CPU | small enough that CPU latency is fine |
| Anything, no accelerator | CPU | always works |

One caveat worth knowing up front, because it is easy to lose an afternoon to:
**an agent defaults to GPU, and the NPU needs a model built for it.** No accelerator executes tools -- the device generates text, and the agent loop parses the tool call out of it and runs the Python function itself. The NPU serves LLMs perfectly well, including tool-calling ones. What it will not do is compile a *group-quantised* export: it needs channel-wise symmetric INT4, which is why OpenVINO publishes a separate [`-int4-cw-ov` family](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu). Point OVMS at a stock `-int4-ov` model with `--target_device NPU` and you get `[NPU_VCL] Compilation failed (0x78000004)` before a single token, which reads like a broken device and is really a mismatched file. GPU is the default because it is correct for every export.

---

## 4. Install and first run

```bash
pip install ovat
ovat setup
```

That is the whole install. `ovat setup` works out which operating system you
are on - and on Linux, which distribution - downloads the matching OpenVINO
Model Server archive, checks its SHA-256, and unpacks it into `~/.ovat/ovms`,
a folder OVAT already searches. **Nothing is added to your `PATH` and no
environment variable is needed.** Run it once per machine; running it again is
a no-op.

On macOS it tells you there is nothing to install, because Intel publishes
Windows and Linux x86-64 builds only, and points you at `ovat chat` - which
runs a model locally with no server at all.

> **Why a subcommand and not the wheel?** The archive is 126–185 MB, Linux
> needs three different builds that cannot be chosen when a wheel is built,
> and macOS has no build at all - so bundling it would charge every Mac user
> ~180 MB for a binary that cannot run. This is the same shape as
> `playwright install`.

If you would rather unpack it yourself, the README has the manual steps. One
warning if you do: take the **`python_on`** build. The C++-only `python_off`
package cannot do tool calling, and the failure is silent - the agent answers
normally and simply never calls a tool.

Then:

```bash
ovat init workflow.yml       # a starter config, with your device auto-detected
ovat doctor workflow.yml     # confirm Python, devices, drivers, OVMS, config
```

`doctor` is worth running before anything else. It is a table, not a wall of
text, and it tells you the specific thing that is wrong:

```
│ Python            │ ✓ ok   │ 3.12.4 (3.10+ required)                       │
│ OpenVINO devices  │ ✓ ok   │ CPU, GPU, NPU                                 │
│ Device routing    │ ✓ ok   │ best here: LLM→GPU emb→NPU whisper→CPU        │
│ OVMS serving      │ ✓ ok   │ ovms via known location (C:\Users\me\ovms)    │
│ Workflow config   │ ✓ ok   │ Qwen3.5-4B-int4-ov on GPU                     │
```

Then a real run:

```bash
ovat serve workflow.yml      # first time: downloads the model, be patient
ovat run workflow.yml -i "what tools do you have?"
```

The first `serve` pulls a 3.5 GB model. It prints elapsed time as it goes and
only gives up after five minutes of no progress at all, so a slow connection is
fine, it will wait as long as the download keeps moving.

**On a Mac?** OVMS is x86-64 only, so there is no server path. But
`openvino_genai` runs natively, and `ovat chat` gives you real local RAG with no
server at all. Development on macOS, serving on the AI PC.

---

## 5. Example 1: Ask your own documents (RAG)

The most useful thing a local agent does: answer from your files, and tell you
which file it used. Nothing is uploaded anywhere.

```bash
# The embedder, once. It is the one model that needs converting.
pip install "ovat[convert]"
optimum-cli export openvino --model BAAI/bge-small-en-v1.5 \
    --task feature-extraction models/bge-small-en-v1.5

ovat index ./examples/rag/docs examples/rag/workflow.yml
ovat serve examples/rag/workflow.yml
ovat run examples/rag/workflow.yml -i "What is OVAT's memory budget?"
```

```
On the minimal tier a full agent run fits in 8 GB. That is the target the project set itself; the number to trust on your own machine is the one `ovat run --telemetry` reports there.

sources: examples/rag/docs/ovat-facts.md
```

The `sources:` line is the point. Retrieval is easy to fake and hard to trust, so
OVAT prints the paths **itself**, from the tool result, not by asking the model
to remember to cite them. Models comply most of the time, which is exactly the
sort of "most of the time" that ruins a demo.

The config that does this is fifteen lines:

```yaml
tools:
  - name: search_docs
    type: builtin

rag:
  embeddings:
    provider: genai                 # runs in-process; indexing needs no server
    model: models/bge-small-en-v1.5
    device: CPU
    dim: 384
  retriever:
    provider: sqlite-vec
    db_path: rag_example.db
  chunk:
    size: 512
    overlap: 64                     # so a sentence across a boundary is findable
```

Every value is a swappable string. `provider: ovms` embeds on the server instead
of locally; `db_path` can live anywhere. Full walkthrough:
[`examples/rag/`](../examples/rag/).

---

## 6. Example 2: Swap the agent framework in one word

If you already use LangChain or LlamaIndex, you should not have to abandon it to
run locally. OVAT supports four engines behind the same config:

```yaml
agent:
  type: react        # native | react | llamaindex | openai-agents
```

- **`native`**. OVAT's own loop. No extra dependencies, and the only engine that
  records per-turn token counts.
- **`react`**. LangChain, `create_agent` + `ChatOpenAI` pointed at OVMS
- **`llamaindex`**. LlamaIndex `FunctionAgent` + `OpenAILike`
- **`openai-agents`**, the OpenAI Agents SDK, against a local server

Same model, same tools, same prompt. And because "it's just a one-word change" is
easy to claim and easy to get wrong, OVAT ships the thing that checks:

```bash
ovat bench examples/react/workflow.yml -i "What can you do?" --out report.json
```

One question, every engine, one server, side by side, build time, answer time,
peak memory, token and tool-call counts. Two deliberate behaviours in that table:

- **A failing engine is a row, not a crash.** A missing extra should not destroy
  the comparison you were running.
- **A dash means unknown, never zero.** If an engine does not report token
  counts the cell is blank, because `0` would read as "used no tokens" and a
  benchmark built on that number is quietly wrong.

You can also override for a single run without touching the file:

```bash
ovat run workflow.yml -i "..." --llamaindex
ovat run workflow.yml -i "..." --langchain      # --react also works
```

Full walkthrough: [`examples/react/`](../examples/react/).

---

## 7. Example 3: Listen and look (audio + vision)

Agents are more useful when they are not text-only. Two more built-in tools:

```yaml
tools:
  - name: transcribe        # speech-to-text, OpenVINO Whisper
    type: builtin
  - name: describe_image    # captions and visual Q&A
    type: builtin
```

```bash
hf download OpenVINO/whisper-base-int8-ov --local-dir models/whisper-base-int8-ov
export OVAT_WHISPER_MODEL=models/whisper-base-int8-ov
export OVAT_VLM_MODEL=models/OpenVINO/Qwen3.5-4B-int4-ov   # the same model!

ovat run examples/audio-multimodal/workflow.yml \
    -i "What is said in examples/audio-multimodal/sample.wav?"
```

The agent decides to call `transcribe`, gets the text, and answers from it. Ask
about an image instead and it calls `describe_image`.

Note the third `export`: the vision model is the **same folder** as the agent's
model. Qwen3.5 is a *unified* export, text generation and image understanding in
one set of weights, so the whole example costs 3.5 GB plus 80 MB of Whisper,
rather than adding a separate 5 GB vision model.

Full walkthrough: [`examples/audio-multimodal/`](../examples/audio-multimodal/).

---

## 8. System requirements

Start from what your machine has, then pick the model that fits it.

| You have | Start with | Expect |
| --- | --- | --- |
| **8 GB RAM** | `Qwen3.5-0.8B-int4-ov` | fast, fine for wiring up tools; weaker answers |
| **16 GB RAM** | `Qwen3.5-4B-int4-ov` (default) | text, vision and tool calling in one model |
| **16 GB + Arc GPU** | same, `device: GPU` | measured 11 s vs 19 s on CPU for the same question |
| **No Intel GPU** | same, `device: CPU` | works; the first question of a session is slow, later ones hit the prefix cache |
| **macOS** | `ovat chat` | development and local retrieval; OVMS has no macOS build |

And the models themselves:

| Model | Download | RAM | For |
| --- | --- | --- | --- |
| `Qwen3.5-0.8B-int4-ov` | 0.9 GB | ~2 GB | 8 GB machines, fast iteration |
| `Qwen3.5-4B-int4-ov` | 3.5 GB | 4.3 GB steady, **6.5 GB peak** | the default; 16 GB machines |
| `Qwen3-8B-int4-ov` | 4.9 GB | ~6-7 GB | strongest text; no vision |

> **These are steady-state figures for a short answer.** A long generation grows
> the KV cache as it goes, and that is not included above. Measured on an AI PC
> during a runaway 18-minute request, the KV cache grew 3.6 → 5.8 GB and
> `ovms.exe` reached **10.6 GB** resident. Cap it with `agent.max_iterations`
> and the model's own answer length if you are near your RAM limit; the steady
> figures are what to plan for, not a ceiling.

The 4B numbers are measured on a real Intel AI PC; the rows written with `~` are estimates from parameter count and precision, not measurements. Note the peak sits **2.2 GB
above** steady state, the load spike decides whether it fits, which is why an
8 GB machine should start at 0.8B even though 4B "looks" like it fits.

Switching tiers is two lines:

```yaml
model:
  name: Qwen3.5-0.8B-int4-ov
  source_model: OpenVINO/Qwen3.5-0.8B-int4-ov
```

One more field matters more than it looks: `tool_parser`. It tells OVMS how to
decode the model's tool calls, and the right answer differs by model family, `qwen3coder` for Qwen3.5, `hermes3` for Qwen3. Get it wrong and the model asks
for a tool, the server cannot read the request, and you get a fluent answer that
called nothing. OVAT derives it from the model name when you leave it out, and
warns loudly if a tool call comes back undecoded.

---

## 9. Where the time and memory actually go

Every run can report what it cost:

```bash
ovat run workflow.yml -i "..." --trace trace.json
```

```json
{
  "engine": "native",
  "model": "Qwen3.5-4B-int4-ov",
  "turns": [
    {"latency_s": 2.14, "finish_reason": "tool_calls",
     "prompt_tokens": 412, "completion_tokens": 38,
     "tool_calls": [{"name": "search_docs", "duration_s": 0.015,
                     "result_chars": 1932,
                     "sources": ["docs/ovat-facts.md"]}]},
    {"latency_s": 1.87, "finish_reason": "stop",
     "prompt_tokens": 2380, "completion_tokens": 64, "tool_calls": []}
  ],
  "totals": {"turns": 2, "latency_s": 4.01, "prompt_tokens": 2792,
             "completion_tokens": 102, "tool_calls": 1,
             "undecoded_tool_call": false, "empty_answer": false}
}
```

Per-turn latency, tokens, which tool ran and for how long, and the sources it
retrieved. `ovat telemetry` adds live CPU, RAM and Intel GPU/NPU numbers.

Because this is a measurement tool, it follows two rules strictly. **Unknown
stays unknown**, a server that reports no token counts produces `null`, never
`0`. And **an unavailable source says why**: on macOS the Intel row reads *"Intel
Unified Telemetry does not run on macOS"* rather than showing zeros, because a
missing sensor and an idle one look identical in a graph.

For deeper observability there is an optional
[plano](https://github.com/katanemo/plano) gateway integration that turns every
request into an OpenTelemetry span, with no OTEL dependency added to OVAT.
See [`examples/plano/`](../examples/plano/).

---

## 10. Extending OVAT with your own tools

The three built-ins are a starting point, not the limit. OVAT speaks
[MCP](https://modelcontextprotocol.io) as a **client**, so any MCP server becomes
a tool:

```yaml
tools:
  - name: filesystem
    type: mcp_stdio
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/docs"]
  - name: my_own_tool
    type: mcp_stdio
    command: ["python", "my_tool.py"]
```

OVAT launches the server, asks what tools it advertises, and hands them to the
agent exactly like built-ins. The agent cannot tell the difference.

The door swings both ways: OVAT's own tools are MCP servers too, so
`python -m ovat.tools.transcribe` gives any MCP-aware agent OpenVINO Whisper.

---

## 11. What it is not

Worth being straight about, so you know whether this fits:

- **Not an inference engine.** OVMS does that. OVAT is the layer above.
- **Not a cloud product.** No hosted anything, no keys, no telemetry phoning
  home.
- **Not a chat UI**, though it ships a terminal one. It is a toolkit for
  *building* things.
- **Not for serving other people's traffic.** It is a local developer tool. There
  is no auth layer, because nothing in it is meant to be reachable from off-box.

---

## 12. Try it

```bash
pip install ovat
ovat init workflow.yml
ovat doctor workflow.yml
```

If `doctor` is green, you are four commands from a working local agent.

- **Repository**, <https://github.com/Lagmator22/ovat>
- **Architecture**, [`docs/ARCHITECTURE.md`](ARCHITECTURE.md), the nine layers and why each exists
- **Examples**, [`examples/`](../examples/), four runnable use cases
- **OpenVINO Model Server**, <https://docs.openvino.ai/2026/model-server/ovms_what_is_openvino_model_server.html>

OVAT is Apache-2.0 and built as
[Google Summer of Code 2026 project #18](https://github.com/openvinotoolkit/openvino)
with the OpenVINO team. Issues and pull requests welcome.
