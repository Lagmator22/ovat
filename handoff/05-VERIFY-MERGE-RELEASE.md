# 05. Verify on the AI PC, merge, release

## A. Owner's manual test on the Intel AI PC (Windows, SSH, cmd.exe)

Repo at `C:\Users\devcloud\ovat`, models at `C:\Users\devcloud\models\OpenVINO`
(`Qwen3-8B-int4-ov`, `Qwen3.6-35B-A3B-int4-ov`). The example workflows ask
for `Qwen3.5-4B-int4-ov`, which is not on that machine, so step 3 makes local
copies that use Qwen3-8B.

`ovat agent` now opens the NEW Vibe-based app. The old app is still there:
`set OVAT_AGENT_UI=classic` before `ovat agent` (and `set OVAT_AGENT_UI=` to go
back). Test both; any difference is a bug to report.

### 1. Get the branch (SSH window 1)
```bat
cd C:\Users\devcloud\ovat
git status
git fetch origin feat/a2a
git switch feat/a2a
git pull --ff-only origin feat/a2a
git log -1 --oneline
.venv\Scripts\python.exe -m pip install -e ".[dev,tui,a2a]"
```
If `git switch feat/a2a` says no such branch: `git switch -c feat/a2a origin/feat/a2a`.
If `git pull` refuses, the local branch has old commits: `git reset --hard origin/feat/a2a`
(only on the test PC, never on a branch with your own unpushed work).

### 2. Automated suite (Python 3.14 on that PC)
```bat
.venv\Scripts\python.exe -m pytest -q -n auto
```
Expect 0 failed. Send the FAILED lines if any.

### 3. Two test workflows that use Qwen3-8B
```bat
powershell -NoProfile -Command "(Get-Content examples\rag\workflow.yml) -replace 'name: Qwen3.5-4B-int4-ov','name: Qwen3-8B-int4-ov' -replace 'source_model: OpenVINO/Qwen3.5-4B-int4-ov','source_model: OpenVINO/Qwen3-8B-int4-ov' -replace 'tool_parser: qwen3coder','tool_parser: hermes3' -replace 'model_repository_path: models','model_repository_path: C:\Users\devcloud\models' | Set-Content rag-test.yml"
powershell -NoProfile -Command "(Get-Content examples\agent\workflow.yml) -replace 'name: Qwen3.5-4B-int4-ov','name: Qwen3-8B-int4-ov' -replace 'source_model: OpenVINO/Qwen3.5-4B-int4-ov','source_model: OpenVINO/Qwen3-8B-int4-ov' -replace 'tool_parser: qwen3coder','tool_parser: hermes3' -replace 'model_repository_path: models','model_repository_path: C:\Users\devcloud\models' -replace 'workflow: examples/rag/workflow.yml','workflow: rag-test.yml' | Set-Content agent-test.yml"
findstr /n "name: source_model tool_parser model_repository_path workflow:" agent-test.yml rag-test.yml
```
Expect `Qwen3-8B-int4-ov`, `hermes3`, `C:\Users\devcloud\models` in both, and
`workflow: rag-test.yml` in agent-test.yml. (Model folders live under
`C:\Users\devcloud\models\OpenVINO\...`; if OVAT cannot find the model, change
the repository path to `C:\Users\devcloud\models\OpenVINO` and the name to
match the folder.)

### 4. Doctor and index
```bat
.venv\Scripts\ovat.exe doctor agent-test.yml
.venv\Scripts\ovat.exe index .\examples\rag\docs rag-test.yml
```
All green, or yellow only for known things. "Embeddings model" must be OK.

### 5. Server and the plain CLI agent
```bat
.venv\Scripts\ovat.exe serve agent-test.yml
.venv\Scripts\ovat.exe run agent-test.yml -i "What is OVAT's memory budget?" --trace run-trace.json
type run-trace.json | findstr /c:"tool_calls" /c:"failed"
```
Expect "OVMS is ready", an answer citing a file in examples\rag\docs, at least
one tool call, `"failed": false`.

### 6. A2A from the command line (SSH window 2)
```bat
cd C:\Users\devcloud\ovat
.venv\Scripts\ovat.exe a2a card rag-test.yml
.venv\Scripts\ovat.exe a2a serve rag-test.yml --port 9100
```
Window 1:
```bat
for /f "delims=" %i in ('.venv\Scripts\ovat.exe a2a token') do set OVAT_A2A_TOKEN=%i
.venv\Scripts\ovat.exe a2a call http://127.0.0.1:9100 "What is OVAT's memory budget?" --token-env OVAT_A2A_TOKEN
set OVAT_A2A_TOKEN=wrong
.venv\Scripts\ovat.exe a2a call http://127.0.0.1:9100 "hello" --token-env OVAT_A2A_TOKEN
```
First call answered (window 2 logs it); bad token refused with a readable error.

### 7. The agent TUI (SSH window 3, terminal at least 80x24)
```bat
cd C:\Users\devcloud\ovat
for /f "delims=" %i in ('.venv\Scripts\ovat.exe a2a token') do set OVAT_A2A_TOKEN=%i
.venv\Scripts\ovat.exe agent agent-test.yml
```

| # | Do | Expect |
|---|---|---|
| 1 | look | OVAT banner (no Mistral/Vibe names), model, device, engine |
| 2 | `What is OVAT's memory budget?` | a search_docs tool row, then an answer citing the file, no `<think>` text |
| 3 | `Ctrl+O` twice | tool details open, then fold |
| 4 | type a second question while #2 runs | queued (shown in the queue), runs after #2 |
| 5 | ask something long, then `Esc` | stops, shown as interrupted |
| 6 | `/help` and `/` + arrows + Enter | command list and menu work; only commands that work are listed |
| 7 | `/tools` | search_docs and delegate with permissions |
| 8 | `/agents` | researcher, team_docs, cloud; "not serving" |
| 9 | `/ask researcher What is OVAT's memory budget?` | subagent answers with a citation |
| 10 | `@researcher What is OVAT's memory budget?` | same, through the @ shortcut |
| 11 | `/ask team_docs What is OVAT's memory budget?` | answered over A2A; window 2 logs it |
| 12 | `/ask cloud Explain KV cache in two lines` | approval panel; keys in the first half second are ignored; choose No: refused |
| 13 | same, choose allow once | allowed; with no API key it says the key is missing (fine) |
| 14 | `/yolo`, repeat 13 | no panel; `/yolo` again turns it off |
| 15 | `Ask the researcher subagent what OVAT's memory budget is.` | the model calls delegate itself; progress shows |
| 16 | `/trace` | turns, tool calls, tokens, seconds of the last answer |
| 17 | `/copy` | last answer on the clipboard |
| 18 | `!dir` | listing, completed |
| 19 | `!ping -n 30 127.0.0.1` then `Esc` | the command stops |
| 20 | `/doctor`, `Ctrl+O`, `/doctor copy` | spinner, problems first, ok checks folded/unfolded, report copied |
| 21 | `/telemetry`, then `?` `r` `c` `Esc` | page fits, nothing shows through, source line on top, NPU and intel rows present (screenshot this); keys work |
| 22 | ask, then `/telemetry` | an agent row with the last answer's tokens |
| 23 | `/engine react`, `/engine llamaindex`, `/engine openai-agents`, ask once each, `/engine native` | each engine answers |
| 24 | `/serve`, `/agents`, `/serve stop` | serving shown, then stopped |
| 25 | `/workflow rag-test.yml` | switches; banner updates |
| 26 | resize very small then big | input stays visible |
| 27 | mouse-select text, `Ctrl+C` | copies, does not quit |
| 28 | `Ctrl+C` `Ctrl+C` | "press again to quit", then quits |
| 29 | `set OVAT_AGENT_UI=classic` and repeat 1-5 | the old app still works |

After Phase 3/4/5 land, add: streaming visible while answering; `/resume`,
`ovat agent --continue`; `/rewind`; `/compact`; `/model`; `/theme`; `/config`
writes; `/mcp`; Down-arrow subagent list and Enter on a child transcript;
parallel subagent group; labelled child approvals; `/agents` dashboard cancel.

### 8. Old launcher TUI still works
```bat
.venv\Scripts\ovat.exe tui
```
`/doctor` Esc, `/telemetry` Esc, `Ctrl+C` twice.

### 9. Clean up
```bat
.venv\Scripts\ovat.exe serve agent-test.yml --stop
del agent-test.yml rag-test.yml run-trace.json
git switch main
```
Ctrl+C in window 2 stops the A2A server.

Most important to report: 21 (NPU/intel telemetry rows), 11 (A2A on Windows),
19 (stopping a shell command on Windows), and the Python 3.14 suite.

## B. Merge order and the conflicts you WILL hit

Merge the small PRs first, one at a time, each only when green:
77, 81, 84, 79, then 80, then 82, then 78 (feat/a2a) last. Never merge 83.

Known conflicts (resolve by MERGING main into the branch, never rebase):
1. **79 vs 80**, `ovat/setup/ovms_installer.py`: 79 adds an ownership
   marker check for `--force`; 80 adds archive-link checks in extraction.
   Different functions, but adjacent hunks and both add imports. Keep both;
   run `tests/test_setup*.py` / installer tests after.
2. **82 vs feat/a2a**, `tests/test_tui.py`: both fixed the same flaky ctrl_c
   tests with an `_until` poll helper. Take ONE copy of the helper (they are
   equivalent), keep both sides' other tests.
3. **82 vs feat/a2a**, `.github/workflows/tests.yml`: 82 adds Python 3.14
   and a weekly schedule; feat/a2a may add test deps or extras. Keep 82's
   matrix plus anything feat/a2a added.
4. After 81 merges, merge main into feat/a2a and re-run the MCP tests (Phase 4
   builds on it).
5. 84 touches `ovat serve` output on Linux; feat/a2a's doctor shows serve
   problems: check the doctor test still passes after merging main.

After each merge into feat/a2a: both suites, then push, then wait for green
CI on all three OSes.

Commit-trailer cleanup: squash-merge 77 and 79 and delete the
`Claude-Session:` line from the squash message in GitHub's merge dialog (see
00 section 4).

## C. Before merging feat/a2a into main (owner decides)

1. Owner's AI PC run of section A, all rows pass.
2. Docs (only now): README section on the new TUI and its keys, `docs/` page
   for the agent TUI (commands, keys, subagents, A2A, telemetry), CHANGELOG.
   Plain English, no em dashes or middle dots.
3. Licensing: the copied Vibe code is Apache-2.0. `NOTICE` at the repo root
   and `ovat/tui/vibe/LICENSE` already exist. NOTICE still says "the adapted
   files are in ovat/tui/ and each one says at the top what was taken": that
   was true before the full copy; the 229 files under `ovat/tui/vibe/` have no
   per-file header. Update NOTICE to name `ovat/tui/vibe/` (copied from
   commit 4ae5c59 of mistralai/mistral-vibe, modified: import paths, Python
   3.10 shims, branding). The owner said he does attribution himself: draft
   it, let him approve.
4. Branding check: `tests/test_tui_branding.py` green; grep the UI strings for
   "Mistral", "Vibe", "Le Chat".
5. Remove nothing of the classic app yet; drop it in a later release after
   users have the new one.

## D. Release

Follow the repo's release workflow (PR 82 hardened it: tag push builds,
tests run first, PyPI trusted publishing). Bump the version in
`pyproject.toml` and the CHANGELOG on a release branch, PR, green, merge, tag
`vX.Y.0` from main. Check `pip install "ovat[tui]"` in a clean venv on
Windows, macOS, Linux, and run `ovat agent --help`.
