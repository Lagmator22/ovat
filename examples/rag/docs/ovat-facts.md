# OVAT engineering notes

## Memory budget

OVAT targets a peak resident set of under 8 GB for a full agent run on the
minimal tier. Peak memory is sampled on a background thread while the run is
in progress. A single reading taken after the run finishes misses the peak,
because Python has already freed the large allocations by then.

## Device routing

The device routing table sends the LLM to the GPU when one is present,
embeddings to the NPU when one is present (otherwise the CPU), and Whisper to
the CPU. The GPU is the default for an agent because it runs any export.

The NPU can run a tool-calling agent too, but only from a channel-wise
symmetric INT4 export, the `-int4-cw-ov` model family. A stock `-int4-ov`
export fails to compile on the NPU before generating a token. The NPU also
caps the prompt length unless OVMS is started with `--max_prompt_len`, and
an agent prompt grows every round.

## The four engines

The `agent.type` field selects one of four engines: native, react,
llamaindex, or openai-agents. All four read the same workflow file and expose
the same tools, so switching between them is a one-word edit. Only the native
engine records per-turn token counts.

## Unknown is not zero

When the server does not report token counts, OVAT records null and displays
a dash. Recording zero instead would read as "used no tokens", and a
benchmark built on that number is quietly wrong.

## Chunk overlap

Indexing shares 64 characters between neighbouring chunks by default. Without
that overlap, a sentence that straddles a chunk boundary is carried in full
by neither chunk, so neither one can answer a question about it.
