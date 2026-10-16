# 06. Paste-ready prompts

Each block below is a complete prompt. Paste it into Claude Code on the Mac,
opened in the `ovat` repo, with this handoff folder available at
`handoff/` on branch `handoff/vibe-port` (or copied anywhere and the path
changed). They work for Opus or Sonnet. Run them one at a time, in order.
Review and test between them.

Usage tips: tell Claude to read ONLY the files the prompt names. Each
prompt already says which. If usage is tight, use Sonnet for the coding
prompts and keep Opus for review.

---

## Prompt 0: orient (cheap, run once per new Claude session)

```
You are continuing work on the OVAT repo (Lagmator22/ovat), branch feat/a2a.
Read handoff/00-START-HERE.md and handoff/01-ARCHITECTURE.md completely.
Do not read the reference/ files yet. Then:
1. git fetch origin && git switch feat/a2a && git pull --ff-only
2. Set up .venv (3.13) and .venv310 (3.10) as 00 section 5 says, if missing.
3. Run both test suites and report pass/fail counts only.
4. Print `git log --oneline -8` and confirm it matches what 00 and 01 describe.
Do not change any code. Reply in under 15 lines.
```

## Prompt 1: Phase 3 (streaming, sessions, rewind, steer, compact, pickers)

```
Continue OVAT's Vibe TUI port on branch feat/a2a. Read, in order:
handoff/00-START-HERE.md, handoff/01-ARCHITECTURE.md,
handoff/02-PHASE3-sessions-streaming.md, and section 3 of
handoff/reference/R2-ovat-capabilities.md. Use
handoff/reference/R5-rpc-method-table.txt to find the method names. The
copied client in ovat/tui/vibe/app_server/ (protocol.py, events.py, models.py,
resources/) is the spec: read it, do not guess.

Do Phase 3 steps 1 to 7 in order. For each step:
- backend change in ovat/agentic/ or ovat/agent/ with a unit test that fails
  without it (prove it by reverting);
- server handler in ovat/tui/server/methods/ with a contract test in
  tests/test_ovat_app_server.py through the real copied client;
- unhide the slash command in ovat/tui/agent_app.py with a pilot test in
  tests/test_vibe_agent_tui.py;
- run tests/test_ovat_app_server.py, then both full suites
  (.venv/bin/python -m pytest -q -n auto and
  .venv310/bin/python -m pytest -q -n 3 -p no:cacheprovider);
- commit with a conventional message whose body says why. Stage files by name.

Hard rules (00 section 3): no Claude attribution of any kind in commits or
code (no Co-Authored-By, no Claude-Session trailer; use git commit -F with a
message file and check `git log --format=%B -1`); never rebase, amend or
force-push; no em dashes or middle dots in any text; do not edit copied Vibe
files under ovat/tui/vibe/ unless unavoidable (then list the edit in the
commit body); do not touch README or docs/ except the generated workflow
reference when a workflow field is added; no new dependencies without asking.

Work on a local branch wip/phase3 from feat/a2a. When all steps are done and
both suites are green, show me: the commit list, test counts, and anything
you could not do and why. Do NOT push until I say so.
```

## Prompt 2: Phase 4 (config, MCP, ask-user, todo, theme, voice, diagnostics)

```
Continue OVAT's Vibe TUI port. Phase 3 is merged into feat/a2a. First merge
origin/main into feat/a2a if PR 81 (fix/mcp-orphan) has been merged (merge,
never rebase), run both suites, and tell me before going on if anything is
red.

Read handoff/00-START-HERE.md, handoff/01-ARCHITECTURE.md,
handoff/03-PHASE4-config-mcp-extras.md, and only the sections of
handoff/reference/R1-vibe-inventory.md and R2-ovat-capabilities.md that file
names. Do Phase 4 steps 1 to 7 (skip step 8 unless I say). Same per-step
process, rules and reporting as the Phase 3 prompt: backend + unit test,
handler + contract test, command + pilot test, both suites, conventional
commit, no attribution, no rebase, no em dashes, no new dependency without
asking me first (step 6 voice needs one: stop and ask). Work on wip/phase4.
Do not push until I say so.
```

## Prompt 3: Phase 5 (subagents and A2A UX)

```
Continue OVAT's Vibe TUI port. Phases 3 and 4 are in feat/a2a.
Read handoff/00-START-HERE.md, handoff/01-ARCHITECTURE.md,
handoff/04-PHASE5-subagents-a2a.md, section 4 of
handoff/reference/R2-ovat-capabilities.md and section 5 of
handoff/reference/R3-subagent-ux-opencode-gemini-hermes.md. Take UX ideas
from opencode, gemini-cli and hermes as described there; do not copy their
code. Vibe's subagent list and transcript widgets are already copied: use
them through child sessions before writing any new widget.

Do Phase 5 steps 1 to 6 (step 7 only if everything else is done). Same
per-step process, rules and reporting as the Phase 3 prompt. Only
ovat/a2a/sdk/ may import the a2a package. Work on wip/phase5. Do not push
until I say so. At the end, run the "Done when" real-model scenario in
handoff/04 on a Linux box if one is available, otherwise list exactly what I
must run on the AI PC.
```

## Prompt 4: review a phase branch (use the strongest model)

```
Review branch wip/<phase> against feat/a2a in the OVAT repo. Read
handoff/00-START-HERE.md section 3 (rules) and handoff/01-ARCHITECTURE.md
sections 4 and 5 (recipe and threading rules). Then read the full diff.
Check, and report each as pass/fail with file:line:
1. any Claude attribution in commit messages or files;
2. any edit under ovat/tui/vibe/ beyond what the commit body declares;
3. any event sent to the client outside the Projector (numbering gaps);
4. any transport write or Projector call from a worker thread;
5. open approvals or question callbacks not resolved on interrupt/close;
6. every new handler has a contract test, every unhidden command a pilot
   test, every fix a test that fails without it (spot-check two by reverting);
7. textual imported outside ovat/tui/ (tests/test_tui_isolation.py);
8. em dashes or middle dots in added text;
9. both suites green on 3.13 and 3.10.
Then list real bugs you found, most severe first. Do not change code.
```

## Prompt 5: merge a phase into feat/a2a and push

```
Branch wip/<phase> was reviewed and approved. In the OVAT repo:
git switch feat/a2a && git pull --ff-only && git merge --ff-only wip/<phase>
If fast-forward is impossible, stop and tell me why. Run both suites. Check
`git log origin/feat/a2a..HEAD --format=%B | grep -i claude` is empty. Then
git push origin feat/a2a and watch CI on PR 78 until every job finishes.
If a job fails, find the root cause, fix it with a new commit (never amend or
force-push), push, repeat until green. Report the final CI state.
```

## Prompt 6: merge the small PRs in order (after the owner's go)

```
In Lagmator22/ovat, merge open PRs into main in this order, one at a time,
only when every CI job is green: 77, 81, 84, 79, 80, 82. Never merge 83
(backup branch) or 78 (feat/a2a). Read handoff/05-VERIFY-MERGE-RELEASE.md
section B first: it lists the conflicts. For 77 and 79 use squash merge and
remove any "Claude-Session:" line from the squash message. When a later PR
conflicts after an earlier merge, merge main into that PR's branch (never
rebase), resolve as section B says, run the suite, push, wait for green, then
merge. After all six, merge main into feat/a2a the same way and get PR 78
green. Report each step.
```
