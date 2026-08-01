# ReAct: the same agent, through LangChain

ReAct is the reason-then-act loop: the model thinks, decides a tool is
needed, calls it, reads the result, and repeats until it can answer.

OVAT implements that loop itself (`agent.type: native`). This example runs
the identical workflow through **LangChain** instead, to show the loop is a
swappable component rather than the product.

```
              ┌─ native         OVAT's own loop.py
workflow.yml ─┼─ react          LangChain          ← this example
  (one file)  ├─ llamaindex     LlamaIndex FunctionAgent
              └─ openai-agents  OpenAI Agents SDK

        all four → the same OVMS server, the same tools
```

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

You get a table of build time, answer time, peak memory, and token/tool
counts where the engine reports them. `report.json` keeps each engine's full
answer, and the complete failure text when one cannot run.

Two deliberate behaviours in that table:

- **A failing engine is a row, not a crash.** One missing extra should not
  destroy the comparison you were running.
- **A dash means unknown, never zero.** If an engine does not report token
  counts, the cell is blank. A `0` would read as "used no tokens".

## Try the switch yourself

Edit one line in `workflow.yml`:

```yaml
agent:
  type: react          # -> native | llamaindex | openai-agents
```

Nothing else changes — not the model, not the tools, not the prompt. Each
engine needs its own extra:

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
- **The framework engines run their own event loop** and refuse to nest
  inside an existing one. That is deliberate, not a limitation to work around.
- **`max_iterations` is a real safety cap.** A model that keeps calling tools
  without converging stops there instead of looping forever.
- **No `rag:` block here**, so `search_docs` answers in stub mode and this
  example needs no downloads beyond the model. Copy the block from
  [`../rag/workflow.yml`](../rag/workflow.yml) for real retrieval.
