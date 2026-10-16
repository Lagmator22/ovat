# 04. Phase 5: subagents and A2A in the new TUI

This is the part the owner cares about most after the port itself: OVAT's
three kinds of helper agent (local workflow subagents, API/cloud subagents,
remote A2A agents) shown and controlled as well as opencode, gemini-cli and
hermes do it.

Read first:
- `reference/R2-ovat-capabilities.md` section 4: exactly what the TUI shows
  today and what the backend already has but throws away (with file:line).
- `reference/R3-subagent-ux-opencode-gemini-hermes.md` section 5: the UX
  design, and 5.8 "top patterns by source file". Licenses are in its
  sections 1.1 / 2.1 / 3.1 / 4.1 (opencode MIT, gemini-cli Apache-2.0,
  hermes MIT, Vibe Apache-2.0). We take IDEAS from opencode/gemini/hermes, not
  code (they are TypeScript); Vibe code we copy.

Local clones used for the research (on the cloud box, may be gone): Vibe at
commit 4ae5c59, opencode, gemini-cli, hermes-agent. If you need a file, fetch
it from GitHub by the path given in R3; do not re-research.

---

## What exists today (do not rebuild)

- `delegate` tool (`ovat/tools/delegate.py`), `SubagentRunner`
  (`ovat/agent/subagents.py`): workflow subagents and API subagents.
- `@name task` in the composer routes to `ask_subagent` (Phase 2a).
- A2A client (`ovat/a2a/sdk/client.py`, `SDKPeer`), `a2a_call` tool, A2A
  server (`/serve`, `ovat/a2a/server.py`, `service.py`), audit log, token in
  `.ovat/a2a/token`, status at `GET /_ovat/status`.
- Bridge: subagent and a2a tool calls render as `SubagentEffectDetail`
  ("Delegating to / Delegated to"); progress replaces one line under the card.
- Rule: only `ovat/a2a/sdk/` may import the `a2a` package.

## Step 1. Child events reach the server (backend)

R2 section 4 "Available but not shown". In `SubagentRunner._forward`
(`ovat/agent/subagents.py`), also re-emit each child bus event as a new
event type `subagent_event {agent, call_id, event: child.to_dict()}` and a
`subagent_done {agent, call_id, trace totals}`. API subagents: emit model,
base URL host, usage. A2A: stop discarding `reply.metadata` in
`a2a_call_impl` and the per-update metadata in `SDKPeer._send`; forward
queue position, remote tool events, final trace, task_id/context_id.
Key progress by call_id, not by "last running tool" (fixes R2 3.12 bullet 4).

## Step 2. Child sessions in the server

Vibe models a subagent as a CHILD SESSION: `ChildSessionUpdated` events and
`ovat/tui/vibe/cli/textual_ui/` `subagent_list.py` / subagent transcripts.
In the Projector, give each delegate call a child session id; route its
`subagent_event`s into that child's history (tool entries, answer). The
parent transcript keeps ONE row per delegate call (no child tool calls in
the parent, except approvals, Step 4). Implement the child-session read
methods the client calls (grep `child` / `subagent` in R5 and in
`ovat/tui/vibe/app_server/resources/`). Then Vibe's existing UI works:
Down from the input goes to the subagent list, Enter opens the read-only
child transcript, Esc returns.

Add child tokens to the session totals (stats).

## Step 3. Parallel group block

When the model issues several delegate calls in one turn, render them as one
group block with counts ("Running 3 agents (1 running, 2 done)") and one
collapsed row each (`icon name | last activity`), ctrl+o expands. Pattern:
gemini-cli `SubagentGroupDisplay.tsx`. This is an OVAT widget (do not edit
copied Vibe widgets; subclass or add a new one under `ovat/tui/widgets/`),
or, preferably, achieved purely by the entries the server sends if Vibe's
widgets already group (check `subagent_list.py` first).

## Step 4. Approvals labelled by agent

Every approval from a child (workflow subagent tools, remote calls) goes to
the ROOT approval bar, titled with the agent name ("subagent `researcher`
wants to run `shell`"). Mark the child `[blocked]` in the list. Remote and
cloud agents: ask before the FIRST call by default; offer once / always for
this agent (session). Show endpoint host and the exact outgoing text, cleaned
by `ovat/tui/server/safe.py:approval_safe`. Patterns: opencode
`routes/session/index.tsx` L197-215, gemini `local-executor.ts`
`messageBus.derive`.

## Step 5. /agents dashboard

A `ModalScreen` (OVAT's own, `ovat/tui/agents_screen.py`), styled like
`ovat/tui/telemetry.py` (ids, `SCOPED_CSS = False`, own background):
sections Local / Cloud / Remote (A2A); for A2A the card (name, skills,
version, endpoint host, auth, reachable or not; keep the `RemoteCard` that
`factory.py` already fetches instead of discarding it). Live rows: status,
elapsed, tool count, tokens. Keys: Enter open transcript, `x` cancel one,
Esc back. Serving state: "A2A :PORT | N peers | M tasks" from
`/_ovat/status`, inbound tasks listed with peer identity and state.

## Step 6. A2A server: approvals over the wire and real cancel

Patterns: gemini-cli `packages/a2a-server/src/agent/task.ts` L542-570 and
L977-1015, `executor.ts` L279.
- When a served tool needs approval, emit `status-update` with state
  `input-required` and metadata `{kind: "tool-call-confirmation", callId,
  tool, args}`; accept a DataPart `{callId, outcome: proceed_once |
  proceed_always | cancel}` to resolve it. Without a reply within a timeout,
  refuse.
- `tasks/cancel` must really stop the running turn (AgentSession interrupt),
  not only mark the task cancelled.
- Keep current safe defaults: loopback bind without a token, audit log,
  inbound text is untrusted.
Tests: drive the server with the a2a client in-process (existing A2A tests
show how; see `tests/test_a2a*.py`).

## Step 7. Steer a running child (optional)

Hermes lets the user type into a running child. For workflow subagents:
queue text into the child's next loop iteration (same slot as Phase 3 steer).
For A2A: a follow-up `message/send` with the same contextId/taskId. Only if
time allows.

## Done when

- Real-model rig run with a workflow that has two subagents and one A2A peer
  (start a second `ovat serve` + `/serve` on another port as the peer):
  parallel delegate shows a group block; Enter opens a child transcript; a
  child's tool approval appears in the root bar with the agent's name;
  cancel from /agents stops the child; a remote approval round-trips via
  input-required.
- Both suites green, contract + pilot tests, screenshots looked at, no
  Claude attribution.
