# The agent TUI, screen by screen

`ovat agent` is a terminal app for working with an agent: you type, it
answers, and everything it does is shown on the screen. This page walks
through the screens, using one run on an Intel AI PC (Qwen3.5-4B on the GPU,
OVMS 2026.4.1). The commands to try the same thing are in
[`examples/agent/`](../examples/agent/).

```bash
pip install "ovat[all]"
ovat serve workflow.yml          # the model server, in its own terminal
ovat agent workflow.yml          # the agent TUI
```

## Start

The banner says which model runs on which device, how many tools and
subagents the workflow has, and which engine drives it. The bar at the bottom
shows live CPU, GPU, NPU and RAM, the tokens used, and `a2a serving` when this
agent is offered to others. Type `/` to open the command menu.

<div align="center">
<img src="assets/screens/agent-tui/hero-agent-start.png" width="860px" alt="The OVAT agent TUI right after start: the banner says OVAT v1.2.0, Qwen3.5-4B-int4-ov on GPU, native engine, 1 tool and 4 subagents. The slash menu is open, and the bar shows live CPU, GPU, NPU and RAM.">
</div>

`/help` lists every key and every command:

<div align="center">
<img src="assets/screens/agent-tui/agent-help.png" width="860px" alt="The /help output: the keys (Enter, Ctrl+J, Esc, Esc Esc to rewind, Ctrl+O, Ctrl+R for voice, Shift+Tab, Ctrl+C) and every slash command from /agents to /yolo.">
</div>

## Subagents

A workflow can name subagents, and the main agent gets one tool, `delegate`,
to hand a whole task to one of them. You can also send a task straight to one
with `/ask <name> <task>` or `@name`. A subagent is one of three kinds:
another OVAT workflow (here `researcher`), an agent reached over A2A (here
`team_docs`), or a model behind an OpenAI-compatible API (`cloud`,
`local_api`).

<div align="center">
<img src="assets/screens/agent-tui/agent-subagent-answers.png" width="860px" alt="Two answers in the agent TUI. The team_docs subagent, reached over A2A, answers that OVAT targets a peak resident set of under 8 GB. The researcher subagent, a workflow, gives the same answer with its source file ovat-facts.md. The list under the prompt shows each subagent and its token count.">
</div>

Each subagent keeps its own conversation. The list under the prompt names
them, and one can be opened read-only to see what it did:

<div align="center">
<img src="assets/screens/agent-tui/agent-researcher-window.png" width="860px" alt="The read-only view of the researcher subagent: its question, the searched files, its cited answer and the note that it is ready for a new instruction. The title line says Viewing Researcher, read-only, Esc to return.">
</div>

`/agents` is the live table of every call: its kind, status, how long it took
and how many tokens it used. `x` cancels one child while the parent carries
on, and `r` refreshes.

<div align="center">
<img src="assets/screens/agent-tui/hero-agent-agents-table.png" width="860px" alt="The /agents screen: the A2A server is serving, four configured subagents, and a table with the team_docs call finished ok in 12.51 seconds with 1663 tokens.">
</div>

## Agent2Agent (A2A)

`/serve` offers the open workflow to other agents over A2A. It starts
`ovat a2a serve` for that workflow in the background, prints the address and
the card URL, and says which tools are exposed. Here `delegate` is not
exposed: a remote caller cannot make your agent hand its work on to a third
one. The token is generated for the project and kept in your own folder, not
in the project. The model server must already be running.

<div align="center">
<img src="assets/screens/agent-tui/hero-agent-serve-and-a2a-answer.png" width="860px" alt="After /serve the A2A server listens on 127.0.0.1 port 9100, the TUI asks it a question through the team_docs subagent and gets a cited answer, and the bar shows a2a serving.">
</div>

The same server can run on its own from a second terminal:

<div align="center">
<img src="assets/screens/agent-tui/a2a-serve-cli.png" width="860px" alt="ovat a2a serve examples/rag/workflow.yml --port 9100: the A2A agent rag is listening on 127.0.0.1 port 9100 for this machine only, with its card address, the token note, the engine and model, the exposed tool search_docs and the audit log path.">
</div>

## Approvals

Anything that leaves the agent, writes a file or runs a command asks first.
The panel says what would happen and where the data would go:

<div align="center">
<img src="assets/screens/agent-tui/agent-approval-local-api.png" width="800px" alt="The approval panel for the local_api subagent: it sends the task to http://localhost:8000/v3 and offers Allow once, Allow for the remainder of the session, Always allow this tool in this project, and Deny.">
</div>

After "Allow once" the model behind the API answers:

<div align="center">
<img src="assets/screens/agent-tui/agent-local-api-pong.png" width="800px" alt="@local_api Say PONG: the subagent answers PONG in 20 tokens.">
</div>

A model that thinks by default (Qwen3 and Qwen3.5 behind OVMS) can spend all
of a small token limit on its reasoning. Set `enable_thinking: false` on the
subagent's `api:` block to turn that off for the call.

`Shift+Tab` switches to auto-approve for the session (the prompt line turns
red); press it again to go back.

<div align="center">
<img src="assets/screens/agent-tui/agent-branch-auto-approve.png" width="800px" alt="After /branch, the TUI says the conversation was branched to a new session and how to resume it with ovat agent --resume. The prompt line is red and reads auto-approve.">
</div>

## See what happened

`/tools` lists the tools and their permission, `/status` the token totals, and
`/trace` the turns of the last answer: how long each took, how many tokens went
in and out, and which tool ran. In this run the model searched the notes twice
and then handed the question to a subagent, which took 53 of the 82 seconds.

<div align="center">
<img src="assets/screens/agent-tui/agent-trace-and-model-picker.png" width="860px" alt="The /tools table with search_docs and delegate, the /status token totals, and the /trace of the last answer, with the /model picker open at the bottom.">
</div>

`/thinking` sets the thinking level for the session:

<div align="center">
<img src="assets/screens/agent-tui/agent-thinking-picker.png" width="860px" alt="The Select Thinking Level picker with Off and High.">
</div>

## Settings

`/config` shows every setting, grouped as popular and advanced, with the
permission of each tool and subagent. Pick a row to edit it. The side panel
says where the value is set (the workflow, or the default), and Ctrl+S saves
the change into `workflow.yml` with your comments kept.

<div align="center">
<img src="assets/screens/agent-tui/hero-agent-config-settings.png" width="860px" alt="The /config settings screen: active model, device, engine, enable_thinking and theme, then temperature, max_tokens, request_timeout, max_iterations, the system prompt and the permission of each tool and subagent.">
</div>

The system prompt opens in a text box. The device is one of CPU, GPU or NPU,
and the engine is one of the four OVAT supports:

<div align="center">
<img src="assets/screens/agent-tui/agent-config-system-prompt.png" width="420px" alt="Editing the system prompt in the settings screen, with the side panel showing where it is set and Ctrl+S to save.">
<img src="assets/screens/agent-tui/agent-config-device.png" width="420px" alt="The device picker in the settings screen: CPU, GPU or NPU, with the workflow value GPU and the default CPU.">
<img src="assets/screens/agent-tui/agent-config-engine.png" width="420px" alt="The engine picker in the settings screen: native, react, llamaindex or openai-agents.">
</div>

## Telemetry

`/telemetry` opens the live numbers: NPU use, KV cache, Intel package power,
CPU and RAM, and the numbers of the last agent answer. Here the model is
answering, so power and CPU load are up:

<div align="center">
<img src="assets/screens/agent-tui/hero-agent-telemetry-load.png" width="860px" alt="The /telemetry screen while a model answers: cards for NPU, KV cache, power 23.5 W, CPU and RAM, and a table with the Intel package and GPU power rows and per-core CPU load.">
</div>

After a run, the same page includes the agent's own numbers: tokens in and
out, turns, tool calls and the latency of the last answer.

<div align="center">
<img src="assets/screens/agent-tui/agent-telemetry-agent.png" width="860px" alt="The /telemetry screen after an answer, with the agent rows: completion tokens, prompt tokens, latency, turns and tool calls, next to the OVMS, NPU and Intel rows.">
</div>

## Health checks

`/doctor` runs the same checks as `ovat doctor` and prints them in the
conversation:

<div align="center">
<img src="assets/screens/agent-tui/agent-doctor.png" width="860px" alt="The /doctor output inside the agent TUI: twelve checks ok, from Python and the framework engines to the OpenVINO devices CPU, GPU and NPU, OVMS serving and OVMS reachable.">
</div>

From a plain terminal, `ovat doctor workflow.yml` shows the same table:

<div align="center">
<img src="assets/screens/agent-tui/cli-doctor.png" width="860px" alt="ovat doctor demo-agent.yml: twelve checks ok and All clear.">
</div>
