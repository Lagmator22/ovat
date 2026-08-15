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

The four sections, plus one setting.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `model` | section | **required** | - | The model section. **Required.** |
| `tools` | list[section] | empty | - | Tools the agent may call. Omit for a tool-less agent. |
| `agent` | section | defaults | - | How the loop behaves. |
| `rag` | section \| null | `null` | - | Vector search for `search_docs`. Omit and the tool answers in a documented stub mode. |
| `model_search_paths` | list[string] | empty | - | Extra folders to scan for local model exports, before `./models` and `~/models`. |

## `model`

Which model to run, where, and how to reach it.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `name` | string | **required** | - | Model the server serves, e.g. `Qwen3.5-4B-int4-ov`. |
| `provider` | string | `ovms` | - | `ovms` talks to the server and can call tools; `genai` runs in-process and cannot. |
| `device` | string | `CPU` | - | Where the model runs. `NPU` needs a channel-wise INT4 export; see the README. |
| `ovms_url` | string | `http://localhost:8000/v3` | - | OpenAI-compatible endpoint. Ends in `/v3` unless a gateway sits in front. |
| `ovms_port` | integer | `8000` | - | Port `ovat serve` binds OVMS to. |
| `tool_parser` | string \| null | `null` | - | How OVMS decodes tool calls. `qwen3coder` for Qwen3.5, `hermes3` for Qwen3. Derived from `name` when omitted. |
| `reasoning_parser` | string \| null | `null` | - | For thinking models that emit a separate reasoning channel. |
| `source_model` | string \| null | `null` | - | Hugging Face id `ovat serve` downloads on first run. |
| `model_repository_path` | string | `models` | - | Folder OVMS keeps its models in. |
| `request_timeout` | float | `1200.0` | - | Cap on one HTTP request, in seconds. A CPU agent turn can genuinely take minutes. |
| `temperature` | float | `0.0` | - | Sampling temperature, applied by every engine. `0.0` keeps tool calls well-formed. |
| `max_tokens` | integer \| null | `4096` | > 0 | Ceiling on one reply. `null` is unbounded, which lets a model that never stops run until the client gives up. |
| `enable_prefix_caching` | boolean | `true` | - | Reuse KV cache across turns sharing a prefix. A large multi-turn win. |
| `ovms_binary` | string \| null | `null` | - | Where the `ovms` executable is. Rarely needed: `ovat setup` installs somewhere OVAT finds. |
| `ovms_cache_size_gb` | integer \| null | `null` | > 0 | KV cache size in GB. Setting it also makes the cache **static** rather than dynamic. Whole numbers only. |
| `ovms_max_prompt_len` | integer \| null | `null` | > 0 | Longest prompt OVMS accepts. **Needed on NPU**, where the default 1024 is smaller than one agent turn. |

## `tools`

One entry per tool. `tools:` is a LIST.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `name` | string | **required** | - | `search_docs`, `transcribe`, `describe_image`, or any tool an MCP server advertises. |
| `type` | string | `builtin` | - | `builtin` runs in-process; `mcp_stdio` launches an external MCP server. |
| `command` | list[string] \| null | `null` | - | Argv that launches the MCP server. Required when `type: mcp_stdio`. |
| `model` | string \| null | `null` | - | Where this tool's own weights live. `transcribe` and `describe_image` only. |
| `device` | string \| null | `null` | - | Device for this tool's model, separate from the agent's. Omit to let the device router choose. |

## `agent`

Which engine drives the tool loop.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `type` | string | `native` | - | Which engine runs the tool loop. Each non-native one needs its matching extra installed. |
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
    Answer from the user's documents and cite the source path.

rag:
  embeddings:
    provider: genai
    model: models/bge-small-en-v1.5-ov
    dim: 384
  retriever:
    provider: sqlite-vec
    db_path: ovat_index.db
```
