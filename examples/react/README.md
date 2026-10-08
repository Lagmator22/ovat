# ReAct: the same agent, through LangChain

ReAct ("reason, then act") is the usual agent loop: the model thinks, decides
it needs a tool, calls it, reads the result, and repeats until it can answer.

OVAT has its own version of that loop (`agent.type: native`). This example
runs the same workflow through **LangChain** instead, to show that the loop is
a part you can swap, not the product itself.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../docs/assets/diagrams/engines-dark.svg">
  <img src="../../docs/assets/diagrams/engines-light.svg" alt="agent.type in workflow.yml selects one of four engines: native (built in), react (LangChain), llamaindex or openai-agents. All four talk to the same OVMS server with the same model, tools and endpoint.">
</picture>

## Run it

```bash
pip install "ovat[langchain]"

ovat serve examples/react/workflow.yml
ovat run examples/react/workflow.yml -i "What can you do?"
```

The engine in use is printed before the answer, so a demo can show it:

```
engine: LangChain (react)
```

## The claim, tested

"Swapping frameworks is a one-word edit" is easy to write and easy to get
wrong. `ovat bench` sends **one question through every engine against one
server** and prints them side by side:

```bash
ovat bench examples/react/workflow.yml -i "What can you do?" --out report.json
```

You get a table of build time, answer time, peak memory, and token and tool
counts where the engine reports them. `report.json` keeps each engine's full
answer, and the complete error text when one cannot run. `--repeat N` runs
each engine N times and shows how many runs succeeded. Each engine runs in its
own process, so one engine's memory is never counted against the next.

Two deliberate behaviours in that table:

- **A failing engine is a row, not a crash.** One missing extra should not
  destroy the comparison you were running.
- **A dash means unknown, never zero.** If an engine does not report token
  counts, the cell is blank. A `0` would read as "used no tokens".

## Try the switch yourself

The fastest way is a flag, with no file edit. The YAML is **never rewritten**,
so the next run goes back to whatever it says:

```bash
ovat run examples/react/workflow.yml -i "What can you do?" --llamaindex
ovat run examples/react/workflow.yml -i "What can you do?" --native
```

Each engine also answers to its library name, since that is the natural first
guess: `--langchain` is the same as `--react`, and `--openai-sdk` is the same
as `--openai-agents`. Naming two different engines is an error ("Pick one
engine"), not a coin flip.

To change it permanently, edit one line in `workflow.yml`:

```yaml
agent:
  type: react          # -> native | llamaindex | openai-agents
```

Nothing else changes: not the model, not the tools, not the prompt. Each
engine except `native` needs its own extra:

| `agent.type` | Install |
| --- | --- |
| `native` | nothing; it is built in |
| `react` | `pip install "ovat[langchain]"` |
| `llamaindex` | `pip install "ovat[llamaindex]"` |
| `openai-agents` | `pip install "ovat[openai-agents]"` |

## Things worth knowing

- **Every engine derives its tool arguments from the same `SCHEMA`.** A tool
  is defined once and works identically on all four; a second hand-kept
  registry is how a tool ends up working on one engine and crashing on
  another.
- **Only `native` records token counts per turn.** The other engines show a
  dash in that column, which means "unknown", not zero.
- **The framework engines run their own event loop** and refuse to start
  inside one that is already running. That is on purpose, not a limitation to
  work around.
- **`max_iterations` is a real safety cap.** A model that keeps calling tools
  without converging stops there instead of looping forever.
- **No `rag:` block here**, so `search_docs` answers in stub mode and this
  example needs no downloads beyond the model. Copy the block from
  [`../rag/workflow.yml`](../rag/workflow.yml) for real retrieval.
