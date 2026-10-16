# 00. START HERE: OVAT x Mistral Vibe TUI port, handoff

Read this file fully before touching anything. It is written so a fresh
Claude Code session (Opus or Sonnet) can carry on with almost no research.
Files in this folder, in reading order:

| File | What it is |
|---|---|
| 00-START-HERE.md | This file: state of every branch and PR, the rules, environment setup |
| 01-ARCHITECTURE.md | How the ported TUI is wired, where every piece lives, how to add an RPC method |
| 02-PHASE3-sessions-streaming.md | Next work item: streaming, sessions, rewind, queue/steer, compact, model picker |
| 03-PHASE4-config-mcp-extras.md | Config writes, MCP panel, ask-user tool, todo, theme, voice, diagnostics |
| 04-PHASE5-subagents-a2a.md | Subagent and A2A UX in the new TUI (opencode / gemini-cli / hermes patterns) |
| 05-VERIFY-MERGE-RELEASE.md | AI PC manual test checklist, PR merge order and known conflicts, release steps |
| 06-PASTE-PROMPTS.md | Ready-to-paste prompts, one per phase, for Claude Code or a Sonnet subagent |
| reference/R1..R5 | Full research reports (Vibe inventory, OVAT capabilities, subagent UX research, Phase 2 design, RPC method table) |

You only need the reference files when a phase file points you there.

---

## 1. The project in one paragraph

OVAT (OpenVINO Agentic Toolkit, repo `Lagmator22/ovat`, Python, MIT-style
open source, PyPI package `ovat`) gives a tool-calling AI agent on Intel
AI PCs from one YAML workflow file. Models are served by OpenVINO Model
Server (OVMS) or run in-process with OpenVINO GenAI. `ovat agent <workflow>`
opens a Textual terminal UI (TUI). The owner wants the full Mistral Vibe TUI
(Apache-2.0, `mistralai/mistral-vibe`) inside OVAT, wired to OVAT's own agent,
plus first-class subagent and A2A (agent-to-agent protocol) UX. That work is on
branch `feat/a2a`. Nothing goes to `main` until the owner has tested it by
hand on the Intel AI PC and merges it himself.

## 2. Branches and PRs (state at handoff)

All PRs target `main`. CI was green on all of these at last check.

| PR | Branch | What | Notes |
|---|---|---|---|
| 77 | fix/serve-loopback | OVMS binds 127.0.0.1 by default | head commit 4e5d2a3 has a `Claude-Session:` trailer (see section 4) |
| 79 | fix/setup-force-scope | `ovat setup --force` only replaces a folder OVAT installed | head 0f4f216 has a `Claude-Session:` trailer |
| 80 | fix/tar-links | refuse tar links that escape the install folder | conflicts with 79 in `ovat/setup/ovms_installer.py` (see 05) |
| 81 | fix/mcp-orphan | kill an MCP stdio server that never finishes connecting | |
| 82 | ci/release-hardening | Python 3.14 in CI, weekly runs, safer release, docs fixes, flaky ctrl_c test fix | conflicts with feat/a2a in `tests/test_tui.py` and `.github/workflows/tests.yml` |
| 84 | fix/linux-ovms-jinja | say clearly when OVMS on Linux cannot load chat templates (Jinja2 missing) | |
| 78 | feat/a2a | THE BIG ONE: A2A, subagents, full Vibe TUI port, OVAT app server | do NOT merge until the owner's AI PC test passes |
| 83 | backup/pre-vibe-full-port | rollback snapshot of feat/a2a before the Vibe port (commit f007618) | NEVER merge. Close it or leave it. It exists only as a restore point |
| 29 | (fork) add-trustabl-action | third-party PR | not ours, ignore |

Rollback: if the Vibe port is rejected, `feat/a2a` can be reset to
`backup/pre-vibe-full-port` (f007618) by the owner. Tags could not be pushed
(the git proxy refused), so a branch is the backup.

### feat/a2a commit history you care about (newest first)

(The newest Phase 2b commits are listed in 01-ARCHITECTURE.md section 9.)

- 43069d9 test(tui): drive the OVAT app server with the copied client
- c4efde6 feat(tui): an OVAT app server for the copied TUI (Phase 2a)
- 8236cd1 feat(tui): project agent events onto the app server's session
- 300fa8d / 8b88407: py3.10 `datetime.UTC` shim, Windows approval test race fix
- f007618: last commit before the full Vibe copy (= backup branch)
- earlier: A2A client/server, subagents, doctor and telemetry pages rebuilt for the TUI

### Size (for the owner's question "how much was copied vs edited")

- `ovat/tui/vibe/`: 229 files, about 48,000 lines copied from Mistral Vibe
  (commit 4ae5c59 of mistralai/mistral-vibe). Only 351 lines were edited beyond
  the mechanical `vibe.` -> `ovat.tui.vibe.` import rename; 149 of 227 Python
  files are byte-identical after the rename. Python 3.10 shims live in one file,
  `ovat/tui/vibe/_compat.py`.
- `ovat/tui/server/`: about 3,000 lines of OVAT's own code (the server side).

## 3. Hard rules (from AGENTS.md, plus the owner's own)

Break any of these and the owner will reject the work.

1. **No Claude attribution anywhere that gets pushed.** No `Co-Authored-By: Claude`,
   no `Claude-Session:` trailer, no "Generated with Claude Code" in commits,
   PR bodies, code comments or docs. Check with
   `git log origin/main..HEAD --format=%B | grep -i claude` before every push.
   If your Claude Code harness adds trailers automatically, remove them
   (write the commit message with `git commit -F file`).
2. Work on feature branches only. Nothing is pushed to `main`.
3. Never rebase, amend or force-push a branch that is already pushed. Fix a
   mistake with a new commit.
4. Stage files by name (`git add path/to/file`), never `git add .` or `-A`.
5. Every fix has a test that FAILS without the fix. Prove it: revert the fix,
   run the test, see red, restore.
6. Public writing (docs, comments, commit messages, PR text) is plain simple
   English. No em dashes, no middle dot as a separator.
7. Verify against primary sources (the actual code, the actual library source
   in the venv) before claiming how something works.
8. Merge only on green CI on every job (Linux, macOS, Windows; Python 3.10,
   3.13, and 3.14 once PR 82 lands).
9. Isolation contract: the CLI must work without the `tui` extra installed.
   No module-level `import textual` outside `ovat/tui/`. `tests/test_tui_isolation.py`
   enforces it.
10. Do NOT update user docs (`docs/`, README) on feat/a2a yet. Docs come after
    the owner accepts the TUI (see 05).
11. Copy Vibe code faithfully. Do not restyle or "improve" copied files. Put
    OVAT behaviour in OVAT's own files (`ovat/tui/server/`, `ovat/tui/agent_app.py`,
    `ovat/tui/agent_commands.py`). The owner adds the Apache-2.0 NOTICE and
    attribution himself before release.
12. A new field in `ovat/config/workflow.py` needs a line in
    `scripts/gen_workflow_reference.py`; then regenerate
    `docs/workflow_yaml_reference.md` (a test fails otherwise). This is the one
    docs file you may touch, because a test demands it.
13. Conventional commit messages (`feat(tui): ...`, `fix(server): ...`), body
    says why.

## 4. One open decision for the owner

Commits 4e5d2a3 (PR 77) and 0f4f216 (PR 79) carry a `Claude-Session:` trailer.
Removing it needs a force-push, which rule 3 forbids. Options: (a) squash-merge
those PRs on GitHub and edit the squash message to drop the trailer (simplest,
no force-push needed), or (b) the owner allows a one-time force-push. Recommend
(a). Do not force-push without the owner saying so.

## 5. Environment setup

### On the Mac (owner's dev box) or any Linux box

```bash
git clone https://github.com/Lagmator22/ovat.git && cd ovat
git checkout feat/a2a
python3.13 -m venv .venv            # 3.13 is the main dev version
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q -n auto
```

Python 3.10 must also pass (CI runs it, and the copied Vibe code needed shims):

```bash
uv venv -p 3.10 .venv310 && VIRTUAL_ENV=.venv310 uv pip install -e ".[dev]"
.venv310/bin/python -m pytest -q -n 3 -p no:cacheprovider
```

Always run the venv's python, never the system one.

### Real model end to end on Linux (no Intel GPU needed, CPU works)

This is how Phase 2a was verified with a real model. It takes about 2 minutes
after downloads.

1. Install OVMS 2026.4.1 (python_on build) with `ovat setup` or unpack it. The
   workflow's `ovms_binary` must point at `.../ovms/bin/ovms`, not the folder.
2. OVMS python_on on Linux needs Jinja2 importable by the SYSTEM python3.12,
   or every chat request fails with "Chat template not loaded correctly":
   ```bash
   python3.12 -m pip install --target ./pyjinja Jinja2==3.1.6 MarkupSafe==3.0.2
   export PYTHONPATH=$PWD/pyjinja   # set in the shell that starts OVMS
   ```
   (PR 84 makes OVAT say this clearly instead of failing silently.)
3. A tiny workflow that exercises a tool call (port 8031 to avoid clashes):
   ```yaml
   model:
     name: Qwen3.5-0.8B-int4-ov
     source_model: OpenVINO/Qwen3.5-0.8B-int4-ov
     model_repository_path: ./models
     ovms_binary: ./ovms/bin/ovms
     device: CPU
     ovms_port: 8031
     ovms_url: http://localhost:8031/v3
     max_tokens: 1024
   tools:
     - name: search_docs
       type: builtin
   agent:
     type: native
     max_iterations: 4
     system_prompt: "You are a helpful assistant. Always call search_docs once before answering."
   ```
4. `ovat serve workflow.yml` then `ovat run workflow.yml "what is OVAT?"`
   (baseline: 2 turns, 1 tool call, about 16 s on a cloud CPU), then
   `ovat agent workflow.yml` for the TUI.

### On the Intel AI PC (Windows, owner tests over SSH with cmd.exe)

Models are at `C:\Users\devcloud\models\OpenVINO` (contains
`Qwen3-8B-int4-ov` and `Qwen3.6-35B-A3B-int4-ov`). Full command list in
05-VERIFY-MERGE-RELEASE.md.

## 6. What is DONE on feat/a2a

- A2A client and server, subagents (`delegate` tool, `@name task`), remote agents.
- Doctor page (`/doctor`): spinner while checks run, problems first, passing
  checks folded, `/doctor copy`. Telemetry page (`/telemetry`): live tiles,
  source line, table, Esc/r/c/? keys, agent trace rows. Both rebuilt to look
  right inside the Vibe-styled app (see `ovat/tui/telemetry.py` docstring).
- Phase 1: the whole Vibe TUI copied to `ovat/tui/vibe/`, importable on 3.10
  to 3.14, branded OVAT.
- Phase 2a: `ovat/tui/server/`, an OVAT implementation of Vibe's app-server
  JSON-RPC protocol over `ovat.agentic.session.AgentSession`. 40 of the 110
  client methods answer; the rest return "not supported in OVAT: <method>".
  70 contract tests drive it through Vibe's real client.
- Phase 2b: `ovat agent` now opens the Vibe app (`OvatAgentApp`) talking to the
  OVAT server. Details and status in 01-ARCHITECTURE.md section 9.

## 7. What is LEFT (in order)

1. Phase 3: streaming tokens, sessions (list/resume/continue/rename/delete/fork,
   autosave), rewind, queue + steer, compact, context window, model and
   thinking pickers, retry. File 02.
2. Phase 4: config writes to workflow.yml, MCP panel, ask-user question tool,
   todo tool, theme, voice (local whisper), diagnostics. File 03.
3. Phase 5: subagent and A2A UX (child sessions, transcripts, parallel group
   block, labelled approvals, /agents dashboard, A2A input-required and real
   cancel). File 04.
4. Phase 6: owner's AI PC test, docs, NOTICE, merge, release. File 05.

Each phase is its own set of commits on `feat/a2a` (or a `wip/<phase>`
branch merged into feat/a2a by fast-forward after review). Each must keep both
test suites green and add contract tests in `tests/test_ovat_app_server.py`.

## 8. How to keep usage low

- Do not re-research. The reference files already answer "what does Vibe
  have", "what does OVAT's backend have", and "how do opencode / gemini-cli /
  hermes do subagents".
- Read the copied Vibe client for the exact contract of a method: the params
  and response models are in `ovat/tui/vibe/app_server/protocol.py`, events in
  `events.py`, history entries in `models.py` and `_effect_models.py`. That is
  the spec. Do not guess.
- Work one method group at a time: add the handler, add a contract test,
  run only `tests/test_ovat_app_server.py` until green, then the full suites.
- A Sonnet subagent can do a whole phase from the prompt in 06 if given this
  folder.
