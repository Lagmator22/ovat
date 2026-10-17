# The agent TUI, subagents and A2A

`ovat agent` is a chat-first TUI: the Intel mark at the top, live tool
lines while the agent works, an approval panel when something wants to
leave your machine, `!` for shell commands and `/` for commands. It needs
the TUI extra, and the A2A parts need the A2A extra:

```powershell
pip install "ovat[tui,a2a]"
```

## On an AI PC (Windows, PowerShell)

Run everything from the repository root. First terminal, the model server:

```powershell
ovat serve examples/agent/workflow.yml
```

Second terminal, a second OVAT agent that others can ask over A2A:

```powershell
ovat index ./examples/rag/docs examples/rag/workflow.yml
ovat a2a serve examples/rag/workflow.yml --port 9100
```

Third terminal, the agent TUI. `team_docs` needs the token the A2A server
made, and `cloud` needs an API key (skip it and that subagent just says so):

```powershell
$env:OVAT_A2A_TOKEN = (ovat a2a token)
$env:MISTRAL_API_KEY = "your key"
ovat agent examples/agent/workflow.yml
```

Things to try:

| Type | What happens |
| --- | --- |
| `What is OVAT's memory budget?` | `search_docs` runs; the answer cites the file |
| `/ask researcher What is OVAT's memory budget?` | the researcher subagent answers in its own context |
| `/ask team_docs What is OVAT's memory budget?` | the question goes over A2A to the second terminal |
| `/ask cloud Explain KV cache in two lines` | an approval panel appears first: 1 once, 2 session, 3 no |
| `!dir` | runs in the shell; Esc stops a long one |
| `/agents`, `/tools`, `/trace` | what the agent has, and what the last answer did |
| `/serve` | offers this workflow over A2A from inside the TUI |

## On a Mac (no OVMS)

A local model runs in-process. It has no tools, but the TUI, `!` commands
and a subagent all work:

```bash
ovat agent examples/agent/local-mac.yml
```

Then try `/ask helper What is 7 times 6?`.

## Permissions

Each tool and subagent is `allow`, `ask` or `deny`. Anything that sends data
off the machine asks by default: an `api` subagent, and an A2A agent that is
not on this PC. In `ovat run` there is nobody to ask, so `ask` means no and
the model is told why. Set `permission: allow` in the workflow to change
that on purpose.
