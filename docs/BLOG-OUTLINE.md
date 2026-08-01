# Blog post outline — OVAT

**Status:** draft outline for review (Ravi asked for this in the 2026-08-01
feedback). Nothing here is published yet.

**Working title options**

1. *One YAML, one command: building tool-calling agents on an Intel AI PC*
2. *Your agent framework is a config file* — OVAT and OpenVINO Model Server
3. *From 50 lines of boilerplate to 12 lines of YAML*

**Audience:** developers who have heard of agents, own or are evaluating an
Intel AI PC, and want local inference without wiring an agent loop by hand.
Assumes Python, assumes no OpenVINO knowledge.

**Length:** 1,500–2,000 words. One diagram, three code blocks, one table.

**The one sentence it must land:** *the agent loop is infrastructure, and
infrastructure belongs in config, not in every project's `main.py`.*

---

## 1. Hook — the boilerplate everyone rewrites (200 words)

Open with the code, not the pitch. Show the ~50 lines every tool-calling
agent needs against OVMS: build the client, hand-write each tool's JSON
schema, call, check `finish_reason`, dispatch the tool, append the result,
loop, guard the iteration count, manage history.

Then the point: none of that is your product. Every project copy-pastes it
and every copy diverges.

> **Show:** the "without OVAT" block from the README, trimmed to ~15 lines.

## 2. The reframe — what if that were config? (250 words)

The same agent as a 12-line `workflow.yml`. Emphasise what is *absent*: no
loop, no schemas, no history management.

Then the payoff that makes it more than sugar — **moving from a 16 GB GPU box
to an 8 GB CPU laptop is a three-line edit.** The agent code never moves;
only the YAML does.

> **Show:** `workflow.yml` next to `minimal.yml`, diffed.

## 3. One word, four engines (300 words)

`agent.type` selects `native`, `react` (LangChain), `llamaindex`, or
`openai-agents`. Same file, same tools, same server.

This is the section that earns trust, because it is falsifiable — and OVAT
ships the falsifier:

```bash
ovat bench examples/react/workflow.yml -i "What can you do?" --out report.json
```

One question, every engine, one server, side by side: build time, answer
time, peak memory, tokens.

Two design choices worth a sentence each, because they signal seriousness:

- **A failing engine is a row, not a crash.** A missing extra should not
  destroy the comparison you were running.
- **A dash means unknown, never zero.** A `0` reads as "used no tokens", and
  a benchmark built on that is quietly wrong — the worst kind.

> **Show:** the four-engine bench table from the AI PC.

## 4. It runs on your machine, not someone's GPU cluster (300 words)

The local story: OVMS on the AI PC for GPU/NPU, `openvino_genai` in-process
for a dev laptop with no server at all. Same config file drives both.

Concrete numbers beat adjectives:

| | Download | RAM |
| --- | --- | --- |
| `Qwen3.5-4B-int4-ov` | 3.5 GB | ~5 GB |
| `Qwen3.5-0.8B-int4-ov` | 0.9 GB | ~2 GB |
| `whisper-base-int8-ov` | 0.08 GB | small |

Land the privacy point plainly: the documents, the audio, and the images
never leave the machine.

> **Show:** `ovat run --trace` output with real token and peak-RSS numbers.

## 5. Three things it can actually do (350 words)

One short subsection each, each with the command and the shape of the answer.

- **RAG** — answers cite the file they came from. The citation is the point:
  it separates a retrieved answer from a plausible one.
- **ReAct** — the model decides a tool is needed, calls it, reads the result,
  answers.
- **Audio + vision** — transcribe a `.wav`, describe an image, in one agent.

The detail worth dwelling on: **Qwen3.5 is a unified model.** Text, vision
and tool calling in one export, so all three examples share a single 3.5 GB
download instead of needing a separate ~5 GB vision model.

> **Show:** the three `examples/` folders.

## 6. The part I got wrong (300 words)

**This is the section that makes the post worth reading.** Most project posts
only show the happy path.

The honest story: it worked on my laptop, and a reviewer could not install
it. What that actually turned out to be —

- The README said `pip install ovat`. The package was never published; that
  command could only ever fail.
- The quickstart never said `git clone`, but step 1 was `pip install -e "."`.
- `doctor` ran *before* `init` created the file it validates, so a new user's
  very first command ended in a red ✗.
- Both the README and OVAT's own generated config told users to run
  `optimum-cli`. Nothing installed it.
- OVMS ships two builds, and the C++-only one **cannot do tool calling** —
  the wrong archive gives you an agent that answers normally and silently
  never calls a tool.

The lesson, stated once and not laboured: *"works on my machine" is not a
property of the code, it is a property of the instructions.* A quickstart is
a program that runs on a stranger's computer, and it deserves the same
skepticism as code — including being executed, in order, on a clean machine.

## 7. Close — what's next (150 words)

Where it is going: OpenTelemetry through the plano gateway, deeper OVMS
integration testing, a published package. Link the repo, the docs, and the
GSoC project page. Invite issues.

---

## Assets to prepare

- [ ] Bench table screenshot from the AI PC (four engines, real numbers)
- [ ] TUI screenshot or GIF — the streaming chat screen is the most visually
      striking thing in the project
- [ ] `--trace` JSON excerpt with genuine token counts
- [ ] The architecture diagram from `docs/ARCHITECTURE.md`
- [ ] A clean-machine install recording, to prove section 6's fixes

## Things to check before publishing

- [ ] Every command in the post run verbatim on a clean machine
- [ ] Model download sizes re-checked (they change)
- [ ] Mentors (Freddy Chiu, Ravi Panchumarthy) reviewed
- [ ] Intel / OpenVINO branding and attribution correct
- [ ] Decide the venue: Intel DevHub, Medium, or the OpenVINO blog
