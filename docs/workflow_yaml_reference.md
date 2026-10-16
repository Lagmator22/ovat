# workflow.yml reference

Every field OVAT accepts, with its type, default and constraints.

> **Generated from the schema** by `scripts/gen_workflow_reference.py`.
> Types, defaults and constraints are read from `ovat/config/workflow.py`,
> so they cannot drift from the code. Edit that file, then re-run the
> script. `tests/test_docs.py` fails if this file is stale or if a field
> is missing a description.

Unknown keys are **errors**, not ignored: a typo fails immediately and
by name rather than silently leaving a default in charge.

---

## `(top level)`

The six sections, plus one setting.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `model` | section | **required** | - | The model section. **Required.** |
| `tools` | list[section] | empty | - | Tools the agent may call. Omit for a tool-less agent. |
| `agent` | section | defaults | - | How the loop behaves. |
| `rag` | section \| null | `null` | - | Vector search for `search_docs`. Omit and the tool answers in a documented stub mode. |
| `model_search_paths` | list[string] | empty | - | Extra folders to scan for local model exports. Searched first, ahead of `OVAT_MODELS`, `./models` and `~/models`, so a workflow that says where its models live wins over the shell. |
| `a2a` | section \| null | `null` | - | How `ovat a2a serve` offers this workflow to other agents. Omit to serve with safe defaults. |
| `subagents` | list[section] | empty | - | Specialists the agent may hand a whole task to, through one `delegate` tool. Used by `ovat agent`. |

## `model`

Which model to run, where, and how to reach it.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `name` | string | **required** | - | Model the server serves, e.g. `Qwen3.5-4B-int4-ov`. |
| `provider` | string | `ovms` | - | `ovms` talks to the server and can call tools; `genai` runs in-process and cannot. |
| `device` | string | `CPU` | - | Where the model runs. `NPU` needs a channel-wise INT4 export; see the README. |
| `ovms_url` | string | `http://localhost:8000/v3` | - | OpenAI-compatible endpoint. Ends in `/v3` unless a gateway sits in front. |
| `ovms_port` | integer | `8000` | - | Port `ovat serve` binds OVMS to. |
| `tool_parser` | string \| null | `null` | - | How OVMS decodes tool calls. `qwen3coder` for Qwen3.5, Qwen3.6 and Qwen3-Coder, `hermes3` for Qwen3, `phi4` for Phi-4-mini, `gptoss`, `llama3` for Llama-3.2, `devstral`. Derived from `name` when omitted; `hermes3` for a family OVAT does not recognise. |
| `reasoning_parser` | string \| null | `null` | - | For thinking models that emit a separate reasoning channel. |
| `enable_thinking` | boolean \| null | `null` | - | Turn a thinking model's reasoning on or off via its chat template. Unset leaves the template's default. |
| `source_model` | string \| null | `null` | - | Hugging Face id `ovat serve` downloads on first run. |
| `model_repository_path` | string | `models` | - | Folder OVMS keeps its models in. |
| `request_timeout` | float | `1200.0` | - | Cap on one HTTP request, in seconds. A CPU agent turn can genuinely take minutes. |
| `temperature` | float | `0.0` | - | Sampling temperature, applied by every engine. `0.0` keeps tool calls well-formed. |
| `max_tokens` | integer \| null | `4096` | > 0 | Ceiling on one reply. `null` is unbounded, which lets a model that never stops run until the client gives up. |
| `top_p` | float \| null | `null` | > 0, <= 1 | Nucleus sampling cut-off. Unset sends nothing. The Qwen3.5 card suggests 0.8 without thinking. |
| `top_k` | integer \| null | `null` | >= -1 | Sample from the k most likely tokens; `-1` for all. Unset sends nothing. |
| `min_p` | float \| null | `null` | >= 0, <= 1 | Drop tokens below this fraction of the top token's probability. Unset sends nothing. |
| `presence_penalty` | float \| null | `null` | >= -2, <= 2 | Discourages repetition. The Qwen3.5 card suggests 1.5 against endless loops. Unset sends nothing. |
| `seed` | integer \| null | `null` | >= 0, <= 4294967295 | Fixed seed, so a sampled run repeats. Unset means a random seed. |
| `enable_prefix_caching` | boolean | `true` | - | Reuse KV cache across turns sharing a prefix. A large multi-turn win. |
| `ovms_binary` | string \| null | `null` | - | Where the `ovms` executable is. Rarely needed: `ovat setup` installs somewhere OVAT finds. |
| `ovms_cache_size_gb` | integer \| null | `null` | > 0 | KV cache size in GB. Setting it also makes the cache **static** rather than dynamic. Whole numbers only. |
| `ovms_max_prompt_len` | integer \| null | `null` | > 0 | Longest prompt OVMS accepts. On `NPU` an unset value becomes **4096**, because OVMS's own NPU cap of 1024 is smaller than one agent turn; set it to go lower. Left to OVMS on CPU and GPU. |
| `ovms_tool_guided_generation` | boolean | `false` | - | Make OVMS hold each tool call to the tool's schema while generating. Read by `ovat serve`. |
| `context_window` | integer \| null | `null` | > 0 | Model context window in tokens, shown by the TUI context meter. Unset reads `max_position_embeddings` from a local model folder (or `ovms_max_prompt_len` on `NPU`). |

## `tools`

One entry per tool. `tools:` is a LIST.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `name` | string | **required** | - | `search_docs`, `transcribe`, `describe_image`, or any tool an MCP server advertises. |
| `type` | string | `builtin` | - | `builtin` runs in-process; `mcp_stdio` launches an external MCP server; `a2a` asks another agent over A2A (needs `pip install 'ovat[a2a]'`). |
| `command` | list[string] \| null | `null` | - | Argv that launches the MCP server. Required when `type: mcp_stdio`. |
| `env` | string \| string \| null | `null` | - | Extra environment for an `mcp_stdio` server, which otherwise sees only HOME, PATH and a few more. `${VAR}` is expanded from your shell. |
| `model` | string \| null | `null` | - | Where this tool's own weights live. `transcribe` and `describe_image` only. |
| `device` | string \| null | `null` | - | Device for this tool's model, separate from the agent's. Omit to let the device router choose. |
| `url` | string \| null | `null` | - | Base URL of the other agent. Its Agent Card is read from there. Required when `type: a2a`. |
| `token_env` | string \| null | `null` | - | NAME of the environment variable holding the bearer token for that agent. Never the token itself. |
| `timeout` | float \| null | `null` | > 0 | Seconds to wait for the other agent's answer. Unset: 900. |
| `description` | string \| null | `null` | - | What the model is told this agent does. Unset: built from the agent's card. |
| `permission` | allow \| ask \| deny \| null | `null` | - | `allow`, `ask` (a person says yes first; with nobody to ask, as in `ovat run`, it does not run) or `deny`. Unset: `ask` for an `a2a` tool on another machine, `allow` otherwise. |

## `agent`

Which engine drives the tool loop.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `type` | string | `native` | - | Which engine runs the tool loop: `native` (built in), `react` (LangChain), `llamaindex`, or `openai-agents`. Each non-native one needs its matching extra, e.g. `pip install 'ovat[langchain]'`. |
| `max_iterations` | integer | `10` | - | Safety cap on tool-calling rounds before the loop gives up. |
| `system_prompt` | string \| null | `null` | - | The agent's persona and standing instructions. |

## `rag.embeddings`

Turning text into vectors.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `provider` | string | `genai` | - | `genai` embeds in-process; `ovms` asks the server. |
| `model` | string | `models/bge-small-en-v1.5` | - | Folder holding the embedding model. |
| `device` | string | `CPU` | - | Device for the embedder. Small and static-shaped, so NPU suits it. |
| `dim` | integer | `384` | - | Vector width. **Must match the model** -- bge-small is 384. A mismatch corrupts the index rather than erroring. |

## `rag.retriever`

Where those vectors are kept.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `provider` | string | `sqlite-vec` | - | `sqlite-vec` persists to `db_path`; `memory` keeps nothing after the run. |
| `db_path` | string | `ovat_index.db` | - | Where `sqlite-vec` writes the index. Ignored by `memory`. |

## `rag.chunk`

How a document is sliced before embedding.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `size` | integer | `512` | - | Characters per chunk before embedding. |
| `overlap` | integer | `64` | - | Characters shared with the next chunk, so meaning is not cut in half at a boundary. |

## `subagents`

One specialist per entry. `subagents:` is a LIST.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `name` | string | **required** | - | Short id: lowercase letters, digits, `-`, `_`. |
| `description` | string | **required** | - | What it is good at. The main model reads this to choose it. |
| `workflow` | string \| null | `null` | - | Another OVAT workflow file, run in-process with a fresh conversation per task. |
| `api` | section \| null | `null` | - | A model behind an OpenAI-compatible API, such as a cloud provider. Its data leaves this machine. |
| `a2a` | section \| null | `null` | - | Another agent reached over A2A. |
| `system_prompt` | string \| null | `null` | - | Instructions for an `api` subagent. |
| `permission` | allow \| ask \| deny \| null | `null` | - | `allow`, `ask` or `deny`. Unset: `ask` for `api` and for `a2a` off this machine, `allow` otherwise. |

## `subagents.api`

A subagent behind an OpenAI-compatible API.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `base_url` | string | **required** | - | The API's base URL, e.g. `https://api.mistral.ai/v1`. |
| `model` | string | **required** | - | Model name at that API. |
| `api_key_env` | string \| null | `null` | - | NAME of the environment variable holding the API key. |
| `max_tokens` | integer | `2048` | > 0 | Longest answer it may write. |
| `temperature` | float | `0.0` | >= 0 | Sampling temperature. |
| `timeout` | float | `300.0` | > 0 | Seconds to wait for its answer. |

## `subagents.a2a`

A subagent reached over A2A.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `url` | string | **required** | - | The agent's base URL. |
| `token_env` | string \| null | `null` | - | NAME of the environment variable holding its bearer token. |
| `timeout` | float | `900.0` | > 0 | Seconds to wait for its answer. |

## `a2a`

Agent2Agent. Needs `pip install 'ovat[a2a]'`.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `server` | section | defaults | - | Settings for `ovat a2a serve`. |

## `a2a.server`

How `ovat a2a serve` offers this workflow. Safe by default.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `name` | string \| null | `null` | - | Name on the Agent Card. Unset: the workflow file's name. |
| `description` | string \| null | `null` | - | Text on the Agent Card, which is public. Never taken from `system_prompt`. |
| `skills` | list[section] \| null | `null` | - | Skills on the card. Unset: one per exposed tool, described by the tool itself. |
| `host` | string | `127.0.0.1` | - | Address to listen on. The default is this machine only. |
| `port` | integer | `9100` | > 0, < 65536 | Port to listen on. |
| `public_url` | string \| null | `null` | - | URL the card advertises, when callers reach this server through a tunnel or proxy. |
| `tokens` | string \| string | empty | - | Caller name to the NAME of the environment variable holding that caller's token. Unset: one generated token in `.ovat/a2a/token`. |
| `expose_tools` | list[string] \| null | `null` | - | Tools a remote caller may trigger. Unset: every builtin that takes no file path; MCP tools only when listed. |
| `expose_sources` | names \| full \| none | `names` | - | Citations sent back: file `names`, `full` paths, or `none`. |
| `max_message_chars` | integer | `8000` | > 0 | Longest message accepted. |
| `max_queue` | integer | `4` | >= 0 | Tasks allowed to wait for the model before new ones are declined. |
| `rate_limit_per_min` | integer | `30` | > 0 | Messages per caller per minute. |
| `task_timeout_s` | float | `900.0` | > 0 | Seconds one task may run before it fails. |
| `max_contexts` | integer | `16` | > 0 | Conversations kept in memory at once; the oldest is dropped first. |
| `max_turns_per_context` | integer | `20` | > 0 | Messages in one conversation before the caller must start a new one. |
| `max_hops` | integer | `3` | >= 1 | How many agents deep a request may travel, so two agents cannot call each other forever. |
| `audit_log` | string \| null | `.ovat/a2a/audit.jsonl` | - | JSON Lines file recording every answer that left. `null` turns it off. |

## `a2a.server.skills`

One skill on the Agent Card. `skills:` is a LIST.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `id` | string | **required** | - | Stable id of the skill. |
| `name` | string | **required** | - | Short name shown to people. |
| `description` | string | **required** | - | What the skill does. |
| `tags` | list[string] | `['ovat']` | - | Keywords. The spec requires at least one. |
| `examples` | list[string] | empty | - | Example requests. |

---

## A complete example

```yaml
model:
  name: Qwen3.5-4B-int4-ov
  source_model: OpenVINO/Qwen3.5-4B-int4-ov
  device: GPU
  tool_parser: qwen3coder

tools:
  - name: search_docs
    type: builtin
  - name: transcribe
    type: builtin
    model: models/whisper-base-int8-ov
    device: CPU

agent:
  type: native
  max_iterations: 10
  system_prompt: >-
    Always call search_docs before answering a question about
    the user's files, and cite the source path you used. Call
    transcribe only for audio files. If the documents do not
    contain the answer, say so plainly rather than guessing.

rag:
  embeddings:
    provider: genai
    model: models/bge-small-en-v1.5
    dim: 384
  retriever:
    provider: sqlite-vec
    db_path: ovat_index.db
```
