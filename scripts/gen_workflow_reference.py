#!/usr/bin/env python
"""Generate docs/workflow_yaml_reference.md from the pydantic schema.

WHY THIS IS GENERATED. A reference table is a second copy of the schema, and
this project's own rule is that two copies drift. The first field added after
a hand-written table ships undocumented, and the first default changed after it
ships a table that is confidently wrong -- which is worse than no table, since
a reader has no reason to distrust it.

So the half that drifts -- names, types, defaults, constraints, which fields
are required -- is read from the models themselves. Only PROSE lives here, in
PURPOSE below, because pydantic cannot hold what the `#` comments in
workflow.py say, and those comments are worth more than a `description=`
one-liner would be.

tests/test_docs.py enforces both halves: every schema field must appear in
PURPOSE, and the committed markdown must match what this script produces. A
field added without a line here fails the suite rather than shipping
undocumented.

    python scripts/gen_workflow_reference.py          # rewrite the doc
    python scripts/gen_workflow_reference.py --check  # CI-style, no write
"""
import argparse
import os
import sys
import typing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ovat.config import workflow as w  # noqa: E402

OUTPUT = os.path.join("docs", "workflow_yaml_reference.md")

#: One sentence per field, keyed (section, field). The long WHY belongs in the
#: source comments; this is the short WHAT a reader scanning a table needs.
PURPOSE = {
    ("model", "name"): "Model the server serves, e.g. `Qwen3.5-4B-int4-ov`.",
    ("model", "provider"): "`ovms` talks to the server and can call tools; "
                           "`genai` runs in-process and cannot.",
    ("model", "device"): "Where the model runs. `NPU` needs a channel-wise "
                         "INT4 export; see the README.",
    ("model", "ovms_url"): "OpenAI-compatible endpoint. Ends in `/v3` unless "
                           "a gateway sits in front.",
    ("model", "ovms_port"): "Port `ovat serve` binds OVMS to.",
    ("model", "tool_parser"): "How OVMS decodes tool calls. `qwen3coder` for "
                              "Qwen3.5, `hermes3` for Qwen3. Derived from "
                              "`name` when omitted.",
    ("model", "reasoning_parser"): "For thinking models that emit a separate "
                                   "reasoning channel.",
    ("model", "source_model"): "Hugging Face id `ovat serve` downloads on "
                               "first run.",
    ("model", "model_repository_path"): "Folder OVMS keeps its models in.",
    ("model", "request_timeout"): "Cap on one HTTP request, in seconds. A CPU "
                                  "agent turn can genuinely take minutes.",
    ("model", "temperature"): "Sampling temperature, applied by every engine. "
                              "`0.0` keeps tool calls well-formed.",
    ("model", "max_tokens"): "Ceiling on one reply. `null` is unbounded, "
                             "which lets a model that never stops run until "
                             "the client gives up.",
    ("model", "enable_prefix_caching"): "Reuse KV cache across turns sharing "
                                        "a prefix. A large multi-turn win.",
    ("model", "ovms_binary"): "Where the `ovms` executable is. Rarely needed: "
                              "`ovat setup` installs somewhere OVAT finds.",
    ("model", "ovms_cache_size_gb"): "KV cache size in GB. Setting it also "
                                     "makes the cache **static** rather than "
                                     "dynamic. Whole numbers only.",
    ("model", "ovms_max_prompt_len"): "Longest prompt OVMS accepts. **Needed "
                                      "on NPU**, where the default 1024 is "
                                      "smaller than one agent turn.",

    ("tools", "name"): "`search_docs`, `transcribe`, `describe_image`, or any "
                       "tool an MCP server advertises.",
    ("tools", "type"): "`builtin` runs in-process; `mcp_stdio` launches an "
                       "external MCP server.",
    ("tools", "command"): "Argv that launches the MCP server. Required when "
                          "`type: mcp_stdio`.",
    ("tools", "model"): "Where this tool's own weights live. `transcribe` and "
                        "`describe_image` only.",
    ("tools", "device"): "Device for this tool's model, separate from the "
                         "agent's. Omit to let the device router choose.",

    ("agent", "type"): "Which engine runs the tool loop. Each non-native one "
                       "needs its matching extra installed.",
    ("agent", "max_iterations"): "Safety cap on tool-calling rounds before "
                                 "the loop gives up.",
    ("agent", "system_prompt"): "The agent's persona and standing "
                                "instructions.",

    ("rag.embeddings", "provider"): "`genai` embeds in-process; `ovms` asks "
                                    "the server.",
    ("rag.embeddings", "model"): "Folder holding the embedding model.",
    ("rag.embeddings", "device"): "Device for the embedder. Small and "
                                  "static-shaped, so NPU suits it.",
    ("rag.embeddings", "dim"): "Vector width. **Must match the model** -- "
                               "bge-small is 384. A mismatch corrupts the "
                               "index rather than erroring.",

    ("rag.retriever", "provider"): "`sqlite-vec` persists to `db_path`; "
                                   "`memory` keeps nothing after the run.",
    ("rag.retriever", "db_path"): "Where `sqlite-vec` writes the index. "
                                  "Ignored by `memory`.",

    ("rag.chunk", "size"): "Characters per chunk before embedding.",
    ("rag.chunk", "overlap"): "Characters shared with the next chunk, so "
                              "meaning is not cut in half at a boundary.",

    ("(top level)", "model"): "The model section. **Required.**",
    ("(top level)", "tools"): "Tools the agent may call. Omit for a "
                              "tool-less agent.",
    ("(top level)", "agent"): "How the loop behaves.",
    ("(top level)", "rag"): "Vector search for `search_docs`. Omit and the "
                            "tool answers in a documented stub mode.",
    ("(top level)", "model_search_paths"): "Extra folders to scan for local "
                                           "model exports, before `./models` "
                                           "and `~/models`.",
}

SECTIONS = [
    ("(top level)", w.WorkflowConfig, "The four sections, plus one setting."),
    ("model", w.ModelConfig, "Which model to run, where, and how to reach it."),
    ("tools", w.ToolConfig, "One entry per tool. `tools:` is a LIST."),
    ("agent", w.AgentConfig, "Which engine drives the tool loop."),
    ("rag.embeddings", w.EmbeddingsConfig, "Turning text into vectors."),
    ("rag.retriever", w.RetrieverConfig, "Where those vectors are kept."),
    ("rag.chunk", w.ChunkConfig, "How a document is sliced before embedding."),
]


def type_name(annotation) -> str:
    """A YAML author's name for a Python annotation.

    `str | None` is written as `string | null`, because the reader is writing
    YAML and `None` is not a thing they can type.
    """
    simple = {str: "string", int: "integer", float: "float", bool: "boolean",
              type(None): "null"}
    if annotation in simple:
        return simple[annotation]
    origin = typing.get_origin(annotation)
    if origin is list:
        inner = typing.get_args(annotation)
        return f"list[{type_name(inner[0])}]" if inner else "list"
    args = typing.get_args(annotation)
    if args:                                   # a union
        return " \\| ".join(type_name(a) for a in args)
    # A nested config is a SECTION to whoever is writing the YAML; the Python
    # class name is an implementation detail they never type.
    name = getattr(annotation, "__name__", str(annotation))
    if name.endswith("Config"):
        return "section"
    return name


def constraint(field) -> str:
    """Any bound pydantic is enforcing, as the reader would state it."""
    parts = []
    for item in field.metadata:
        for attribute, symbol in (("gt", ">"), ("ge", ">="),
                                  ("lt", "<"), ("le", "<=")):
            value = getattr(item, attribute, None)
            if value is not None:
                parts.append(f"{symbol} {value}")
    return ", ".join(parts) or "-"


def default_text(field) -> str:
    if field.is_required():
        return "**required**"
    value = field.get_default(call_default_factory=True)
    if value is None:
        return "`null`"
    if value == [] or value == {}:
        return "empty"
    if isinstance(value, w.StrictModel):
        return "defaults"
    # YAML spells booleans lowercase. Printing Python's `True` in a reference
    # a reader copies from is a small lie that produces a real config error.
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    return f"`{value}`"


def render() -> str:
    lines = [
        "# workflow.yml reference",
        "",
        "Every field OVAT accepts, with its type, default and constraints.",
        "",
        "> **Generated from the schema** by `scripts/gen_workflow_reference.py`.",
        "> Types, defaults and constraints are read from `ovat/config/workflow.py`,",
        "> so they cannot drift from the code. Edit that file, then re-run the",
        "> script. `tests/test_docs.py` fails if this file is stale or if a field",
        "> is missing a description.",
        "",
        "Unknown keys are **errors**, not ignored: a typo fails immediately and",
        "by name rather than silently leaving a default in charge.",
        "",
        "---",
        "",
    ]
    for title, model, blurb in SECTIONS:
        lines += [f"## `{title}`", "", blurb, "",
                  "| Field | Type | Default | Constraints | Purpose |",
                  "| --- | --- | --- | --- | --- |"]
        for name, field in model.model_fields.items():
            purpose = PURPOSE.get((title, name), "")
            lines.append(f"| `{name}` | {type_name(field.annotation)} | "
                         f"{default_text(field)} | {constraint(field)} | "
                         f"{purpose} |")
        lines.append("")
    lines += ["---", "",
              "## A complete example", "",
              "```yaml",
              "model:",
              "  name: Qwen3.5-4B-int4-ov",
              "  source_model: OpenVINO/Qwen3.5-4B-int4-ov",
              "  device: GPU",
              "  tool_parser: qwen3coder",
              "",
              "tools:",
              "  - name: search_docs",
              "    type: builtin",
              "  - name: transcribe",
              "    type: builtin",
              "    model: models/whisper-base-int8-ov",
              "    device: CPU",
              "",
              "agent:",
              "  type: native",
              "  max_iterations: 10",
              "  system_prompt: >-",
              "    Answer from the user's documents and cite the source path.",
              "",
              "rag:",
              "  embeddings:",
              "    provider: genai",
              "    model: models/bge-small-en-v1.5-ov",
              "    dim: 384",
              "  retriever:",
              "    provider: sqlite-vec",
              "    db_path: ovat_index.db",
              "```",
              ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed doc is out of date")
    args = parser.parse_args()

    generated = render()
    if args.check:
        try:
            with open(OUTPUT, encoding="utf-8") as handle:
                current = handle.read()
        except OSError:
            print(f"{OUTPUT} is missing; run this script without --check")
            return 1
        if current != generated:
            print(f"{OUTPUT} is out of date; re-run this script")
            return 1
        print(f"{OUTPUT} is current")
        return 0

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        handle.write(generated)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
