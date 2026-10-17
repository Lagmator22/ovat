# Coding agent: work on your own project

`ovat agent` with file, search, shell and web tools. It reads, searches,
edits and tests the code in the folder you start it from. Nothing leaves the
machine unless you allow a web tool.

## What you need

| | |
| --- | --- |
| Model | `OpenVINO/Qwen3.6-35B-A3B-int4-ov`, a 35B mixture-of-experts export (about 3B active per token) |
| Memory | enough for the 35B weights. Check before the first run |
| Server | OVMS on Windows or Linux. macOS has no OVMS |
| Install | `pip install "ovat[tui]"` |

A smaller machine can use a smaller Qwen3.5 export. Change `model.name` and
`model.source_model` in `workflow.yml` and keep the rest.

## Run it

Start from the project you want to work on:

```bash
cd path/to/your/project
ovat serve path/to/ovat/examples/coding/workflow.yml     # start OVMS
ovat agent path/to/ovat/examples/coding/workflow.yml     # the agent TUI
```

The first `serve` downloads the model. Watch `ovms.log` if it looks stuck.

## What asks first

| Tool | Default |
| --- | --- |
| `read_file`, `grep`, `todo`, `ask_user` | runs |
| `write_file`, `edit`, `shell`, `web_fetch`, `web_search` | asks you first |

Each question offers once, for the session, or always. "Always" is saved in
`~/.ovat/permissions.json` for this project, not in the project folder. `Shift+Tab`
approves everything for the session. `ovat run` has nobody to ask, so the
tools that ask do not run there.

File tools stay inside the folder you started from. `shell` runs from that
folder but is not fenced in, so read each command before you allow it.

## Things to try

| Type | What happens |
| --- | --- |
| `Find where the config is loaded and explain it` | `grep` and `read_file` run, no questions |
| `Rename the helper foo to bar and run the tests` | an `edit` card with a diff asks first, then `shell` asks |
| `/todo` | the plan the agent is following |
| `Esc Esc` | go back to an earlier message |
| `ovat agent --continue` | pick up the last session in this folder |

`web_search` needs a provider. The comments in `workflow.yml` show how to
turn one on.
