# RAG: ask questions about your own documents

The agent searches a folder you indexed, then answers **citing the file each
fact came from**. Nothing leaves the machine.

```
your .md/.txt files ──ovat index──> chunks + vectors in sqlite-vec
                                          │
   "What is OVAT's memory budget?" ───────┤
                                          ▼
             agent calls search_docs ──> top chunks + source paths
                                          ▼
                          answer, with the path it used
```

## What you need

| | |
| --- | --- |
| Model | `OpenVINO/Qwen3.5-4B-int4-ov` (3.5 GB), or the 0.8B tier, 0.9 GB |
| Embedder | `bge-small-en-v1.5`, converted once (~130 MB) |
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

Expected shape of the answer: **under 8 GB**, together with the path
`examples/rag/docs/ovat-facts.md`. The citation is the point, it is what
separates a retrieved answer from a plausible one.

More questions the sample documents can answer, and pre-training cannot:

```bash
ovat run examples/rag/workflow.yml -i "Which device is never used for tool calling, and why?"
ovat run examples/rag/workflow.yml -i "Why does OVAT record null instead of 0 for tokens?"
ovat run examples/rag/workflow.yml -i "Which engine records per-turn token counts?"
```

## On macOS (no OVMS)

Indexing works everywhere. For the answering half, use the local path, which
needs no server and does retrieve-then-answer without tool calling:

```bash
hf download OpenVINO/Qwen3.5-0.8B-int4-ov --local-dir models/Qwen3.5-0.8B-int4-ov
ovat index ./examples/rag/docs examples/rag/workflow.yml
ovat chat examples/rag/workflow.yml -i "What is OVAT's memory budget?"
```

## Things worth knowing

- **Re-indexing is cheap, re-converting is not.** The embedder conversion is
  a one-time cost; `ovat index` can be re-run whenever your documents change.
- **`chunk.overlap` is not decoration.** Without it, a sentence that straddles
  a chunk boundary is findable by neither chunk.
- **Leave the `rag:` block out entirely** and `search_docs` still answers, in
  a documented stub mode, useful for testing wiring before downloading
  anything.
- **`dim` must match the embedder.** bge-small emits 384 numbers per chunk;
  a mismatch here is a confusing failure at query time, not at index time.
