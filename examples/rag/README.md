# RAG: ask questions about your own documents

The agent searches a folder you indexed, then answers and **names the file each
fact came from**. Nothing leaves the machine.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../docs/assets/diagrams/rag-dark.svg">
  <img src="../../docs/assets/diagrams/rag-light.svg" alt="RAG in two steps. Index once: files are cut into chunks, turned into 384 numbers each by bge-small, and stored in a sqlite-vec file. Every question: the question is turned into numbers, the closest chunks are found, and the model answers from them and names the source files.">
</picture>

RAG means "retrieval-augmented generation": first find the passages in your
documents that match the question, then let the model answer from them.

## What you need

| | |
| --- | --- |
| Model | `OpenVINO/Qwen3.5-4B-int4-ov` (3.5 GB), or the 0.8B tier, 0.9 GB |
| Embedder | `bge-small-en-v1.5`, a small model that turns text into numbers, converted once (about 130 MB) |
| Server | OVMS on Windows/Linux. macOS: use `ovat chat`, below |

## Run it

```bash
# 1. the embedder. There is no pre-built OpenVINO IR of bge-small, so this
#    conversion is the one unavoidable step. optimum-cli is in the extra.
pip install "ovat[convert]"
optimum-cli export openvino --model BAAI/bge-small-en-v1.5 \
    --task feature-extraction models/bge-small-en-v1.5

# 2. index the sample documents in this folder (swap in your own any time)
ovat index ./examples/rag/docs examples/rag/workflow.yml

# 3. serve, and ask
ovat serve examples/rag/workflow.yml
ovat run examples/rag/workflow.yml -i "What is OVAT's memory budget?"
```

Expected shape of the answer: **under 8 GB**, followed by a line like
`sources: examples/rag/docs/ovat-facts.md`. That line is the point: it is what
separates an answer that came from your files from one that only sounds
right. OVAT prints it itself, from what the search returned, rather than
trusting the model to cite.

More questions the sample documents can answer, and pre-training cannot:

```bash
ovat run examples/rag/workflow.yml -i "What does the NPU need in order to run a tool-calling agent?"
ovat run examples/rag/workflow.yml -i "Why does OVAT record null instead of 0 for tokens?"
ovat run examples/rag/workflow.yml -i "Which engine records per-turn token counts?"
```

## On macOS (no OVMS)

Indexing works everywhere. For the answering half, use the local path. It
needs no server: it always searches first, then answers, and never calls a
tool:

```bash
hf download OpenVINO/Qwen3.5-0.8B-int4-ov --local-dir models/Qwen3.5-0.8B-int4-ov
ovat index ./examples/rag/docs examples/rag/workflow.yml
ovat chat examples/rag/workflow.yml -i "What is OVAT's memory budget?"
```

`ovat chat` finds the model in `models/` by itself; `--model-path` picks one.
There is no answer length cap by default. The answer streams until the model
stops, and Ctrl-C stops it early. `--max-tokens N` sets a cap.

## Things worth knowing

- **Re-indexing is cheap, re-converting is not.** The embedder conversion is
  a one-time cost; `ovat index` can be re-run whenever your documents change.
- **OVAT tells you when the index is out of date.** `ovat index` records when
  it read each file. If you edit or delete one afterwards, `ovat run` and
  `ovat chat` name it in a yellow warning, because an old index answers
  confidently from text that is no longer there. Re-run `ovat index`.
- **`chunk.overlap` is not decoration.** Without it, a sentence that straddles
  a chunk boundary is findable by neither chunk.
- **Leave the `rag:` block out entirely** and `search_docs` still answers,
  with a clearly marked `[stub]` result. That is useful for testing the wiring
  before downloading anything.
- **`dim` must match the embedder.** bge-small makes 384 numbers per chunk.
  A mismatch fails in a confusing way when you ask, not when you index.
