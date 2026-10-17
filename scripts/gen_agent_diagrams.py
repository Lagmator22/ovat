#!/usr/bin/env python3
"""Draw the agent-TUI diagrams: layers, approvals, A2A and subagents.

One spec per diagram makes three things, so they cannot drift apart:
  docs/assets/diagrams/<name>-light.svg and -dark.svg   animated, for GitHub
  a <figure data-seq> block for the docs site (ovat-navigate)

The five older diagrams (core-flow, agent-loop, engines, devices, rag) are
edited by hand; only these four come from here.

    python scripts/gen_agent_diagrams.py                 # write the SVGs
    python scripts/gen_agent_diagrams.py --site-out F    # also the site HTML

Every fact in a diagram is checked against the code or AGENTS.md when it is
written. If a name here stops being true, change the spec and run this again.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "diagrams"

LIGHT = ("--text:#1F2328;--sub:#57606A;--card:#FFFFFF;--card2:#F6F8FA;--border:#D0D7DE;"
         "--line:#8C959F;--accent:#0068B5;--soft:rgba(0,104,181,0.08);--purple:#8F5CFF;"
         "--ok:#3DD68C;--oktext:#1A7F37;--warn:#FFC107;--warntext:#9A6700;"
         "--halo:rgba(0,104,181,0.22);--mono:#0068B5")
DARK = ("--text:#E6EDF3;--sub:#9AA7B4;--card:#161B22;--card2:#0F141A;--border:#30363D;"
        "--line:#6E7681;--accent:#00C7FD;--soft:rgba(0,199,253,0.08);--purple:#A98BFF;"
        "--ok:#3DD68C;--oktext:#3DD68C;--warn:#FFC107;--warntext:#FFC107;"
        "--halo:rgba(0,199,253,0.25);--mono:#00C7FD")

BASE_CSS = """text{font-family:"Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;fill:var(--text)}
.title{font-weight:650}
.mono{font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;fill:var(--mono);font-size:12px}
.sub{fill:var(--sub);font-size:12.5px}
.small{fill:var(--sub);font-size:11.5px}
.num{fill:#fff;font-size:11.5px;font-weight:700;text-anchor:middle}
.box{fill:var(--card);stroke:var(--border);stroke-width:1.2}
.tool .box{stroke:var(--purple)}
.ok .box{stroke:var(--ok)}
.warn .box{stroke:var(--warn)}
.link{fill:none;stroke:var(--line);stroke-width:1.6;stroke-dasharray:6 5;animation:flow 1.1s linear infinite}
.link-ok{fill:none;stroke:var(--ok);stroke-width:1.6;stroke-dasharray:6 5;animation:flow 1.1s linear infinite}
.link-purple{fill:none;stroke:var(--purple);stroke-width:1.6;stroke-dasharray:6 5;animation:flow 1.1s linear infinite}
.link-warn{fill:none;stroke:var(--warn);stroke-width:1.6;stroke-dasharray:6 5;animation:flow 1.1s linear infinite}
.hl{fill:var(--soft);stroke:url(#brand);stroke-width:4}
.arrowhead{fill:var(--line)}
@keyframes flow{to{stroke-dashoffset:-22}}
@media (prefers-reduced-motion: reduce){.fx{display:none}*{animation:none!important}}"""

TONE_VAR = {"": "--accent", "tool": "--purple", "ok": "--ok", "warn": "--warn"}
LINK_CLASS = {"": "link", "tool": "link-purple", "ok": "link-ok", "warn": "link-warn"}


def card(cid, num, x, y, w, h, title, lines, tone=""):
    return dict(id=cid, num=num, x=x, y=y, w=w, h=h, title=title, lines=lines, tone=tone)


def link(lid, d, tone=""):
    return dict(id=lid, d=d, tone=tone)


def step(cards, link_id, text, tone=""):
    return dict(cards=cards, link=link_id, text=text, tone=tone)


# --------------------------------------------------------------------------
# The four diagrams
# --------------------------------------------------------------------------

AGENT_TUI = dict(
    name="agent-tui", w=880, h=372,
    title="The agent TUI, from keystroke to card",
    desc=("You type in the terminal client. It sends a JSON-RPC request to OVAT's app server, "
          "which starts a turn on AgentSession. AgentSession runs the chosen engine on a worker "
          "thread; the engine asks the model on OVMS, and every tool call goes through one wrapper "
          "that reports events. The events travel back up the same layers and the client draws them."),
    header="$ ovat agent workflow.yml",
    cards=[
        card("client", 1, 20, 62, 180, 112, "Terminal client",
             ["~ovat/tui/vibe/", "keys, widgets, cards", "copied, see NOTICE"]),
        card("server", 2, 240, 62, 180, 112, "App server",
             ["~ovat/tui/server/", "JSON-RPC requests,", "turns, approvals"]),
        card("session", 3, 460, 62, 180, 112, "AgentSession",
             ["~ovat/agentic/", "a worker thread per", "turn, one event callback"]),
        card("engine", 4, 680, 62, 180, 112, "Engine and tools",
             ["four engines, one", "wrapper around every", "tool call"]),
        card("model", 5, 680, 238, 180, 82, "Model", ["OVMS (GPU, NPU, CPU)", "or in-process on CPU"]),
    ],
    links=[
        link("l12", "M200 118 L238 118"), link("l23", "M420 118 L458 118"),
        link("l34", "M640 118 L678 118"), link("l45", "M770 174 L770 236"),
        link("back", "M740 176 C740 216 110 216 110 178", "ok"),
    ],
    labels=[(400, 236, "events go back up the same way", "small", "middle")],
    steps=[
        step(["client", "server"], "l12", "You type or press a key. The client turns it into a JSON-RPC request to OVAT's own app server."),
        step(["server", "session"], "l23", "The app server starts a turn on AgentSession. A worker thread runs it, so the screen never waits."),
        step(["session", "engine"], "l34", "AgentSession hands the question to the engine named in workflow.yml: native, react, llamaindex or openai-agents."),
        step(["engine", "model"], "l45", "The engine asks the model on OVMS. The reply is either an answer or a request for a tool.", "tool"),
        step(["engine", "session", "server"], "back", "Every tool call passes one wrapper (events.observed). It checks the permission, reports tool_started and tool_ended, and watches the stop flag. The events travel back up.", "ok"),
        step(["client"], None, "The client draws each event as it arrives: streamed text, a tool card with its diff, an approval panel, the bar at the bottom.", "ok"),
    ],
    foot="ovat/agentic/ imports no UI toolkit (a test scans it), so every layer below the client is tested without a screen.",
    note="The same engines run under ovat run, ovat bench and the A2A server. Only the client is specific to the terminal.",
)

APPROVAL = dict(
    name="approval-flow", w=880, h=470,
    title="A tool call, and who may say yes",
    desc=("The model asks for a tool. One wrapper, events.observed, checks the permission of that tool "
          "or subagent. Allow runs it. Ask opens an approval panel with once, session, always and deny "
          "when someone is there to answer, and counts as no when nobody is. Deny sends the model a plain sentence."),
    header="write_file, edit, shell, web_fetch and web_search ask first",
    cards=[
        card("call", 1, 20, 185, 170, 100, "Tool call", ["the model asks for", "write_file, shell, or", "a subagent"]),
        card("wrap", 2, 230, 185, 190, 100, "Wrapper", ["~events.observed()", "one wrapper for all", "four engines"]),
        card("perm", 3, 460, 185, 170, 100, "Permission", ["allow, ask or deny", "per tool and subagent"]),
        card("run", 4, 690, 20, 170, 90, "allow: runs", ["the result goes to", "the model"], "ok"),
        card("panel", 5, 690, 130, 170, 110, "ask: a person", ["once, session,", "always, or deny.", "The worker waits."], "tool"),
        card("nobody", 6, 690, 260, 170, 90, "ask, no approver", ["ovat run, bench, A2A:", "counts as no"], "warn"),
        card("deny", 7, 690, 370, 170, 80, "deny: refused", ["a sentence goes back", "to the model"], "warn"),
    ],
    links=[
        link("l12", "M190 235 L228 235"), link("l23", "M420 235 L458 235"),
        link("l34", "M630 215 C660 215 660 65 688 65", "ok"),
        link("l35", "M630 235 L688 190", "tool"),
        link("l36", "M630 255 C660 255 660 305 688 305", "warn"),
        link("l37", "M630 270 C650 270 650 410 688 410", "warn"),
    ],
    labels=[],
    steps=[
        step(["call", "wrap"], "l12", "The model asks for a tool. On every engine the call goes through one wrapper, so there is a single place to get this right."),
        step(["wrap", "perm"], "l23", "The wrapper reports tool_started and checks the permission of that tool or subagent."),
        step(["perm", "run"], "l34", "allow: the tool runs and its result goes back to the model.", "ok"),
        step(["perm", "panel"], "l35", "ask, with a screen: the worker waits while the approval panel offers once, for the session, always for this project, or deny. Always is saved in ~/.ovat/permissions.json, never in the project.", "tool"),
        step(["perm", "nobody"], "l36", "ask, with nobody to ask (ovat run, a bench worker, an A2A server): it counts as no, so nothing is sent out unasked.", "warn"),
        step(["perm", "deny"], "l37", "deny: the model gets a plain sentence saying so. A refused tool never shows a traceback.", "warn"),
    ],
    foot="Reading and searching run on their own. Writing, editing, shell and web tools ask first.",
    note="A subagent that sends data off the machine asks too: an api subagent, and an A2A agent that is not on this PC.",
)

A2A = dict(
    name="a2a-flow", w=880, h=390,
    title="One question to another agent over A2A",
    desc=("A caller sends a message. The server checks the host, the body size and the bearer token, then "
          "admission limits, then gives each caller and conversation its own agent. A run takes the conversation "
          "lock and then the one-model gate, runs only the exposed tools, and the answer comes back as a task update."),
    header="$ ovat a2a serve workflow.yml --port 9100        (or /serve in the agent TUI)",
    cards=[
        card("caller", 1, 20, 60, 180, 124, "Caller", ["a subagent of kind a2a", "token sent per request", "to its own host only"]),
        card("guard", 2, 240, 60, 180, 124, "HTTP guard", ["host check, then a", "size cap, then the", "bearer token"]),
        card("admit", 3, 460, 60, 180, 124, "Admission", ["loop and hop limits,", "empty or too long,", "queue full, rate limit"]),
        card("conv", 4, 680, 60, 180, 124, "Conversation", ["one agent per caller", "and conversation id,", "built with the exposed", "tools only"]),
        card("gate", 5, 680, 238, 180, 106, "Slot, then gate", ["one run per chat,", "then the one-model", "gate (a lock)"]),
        card("agent", 6, 460, 238, 180, 106, "Agent runs", ["the engine and tools", "as in the TUI; a person", "can approve a tool"], "tool"),
        card("reply", 7, 240, 238, 180, 106, "Reply", ["answer, sources and", "token counts, as an", "A2A task update"], "ok"),
    ],
    links=[
        link("l12", "M200 122 L238 122"), link("l23", "M420 122 L458 122"), link("l34", "M640 122 L678 122"),
        link("l45", "M770 184 L770 236"), link("l56", "M678 291 L642 291"), link("l67", "M458 291 L422 291", "ok"),
        link("l71", "M238 300 C130 300 110 270 110 186", "ok"),
    ],
    labels=[],
    steps=[
        step(["caller", "guard"], "l12", "The caller sends the message. The bearer token is added to each request, and only for the host it was configured for, never for a host that a remote card names."),
        step(["guard", "admit"], "l23", "The server checks the Host header, caps the body size and checks the token before anything else."),
        step(["admit", "conv"], "l34", "Admission refuses a request that loops back to this server, has passed through too many agents, is empty or too long, or arrives when the queue or the caller's rate limit is full."),
        step(["conv", "gate"], "l45", "Each caller and conversation gets its own agent, so two callers never share a memory."),
        step(["gate", "agent"], "l56", "The run takes its conversation's lock first, then the one-model gate. While a person decides on an approval the gate is given back, so other callers are not blocked.", "tool"),
        step(["agent", "reply"], "l67", "The agent runs only the tools that were exposed. Coding tools, web tools and delegate are not offered to a remote caller.", "ok"),
        step(["reply", "caller"], "l71", "The answer comes back as a task update. The audit log (.ovat/a2a/audit.jsonl) keeps the question and answer and is readable only by you.", "ok"),
    ],
    foot="The A2A SDK is imported in one place only (ovat/a2a/sdk/); a test keeps it that way.",
    note="A task that is not answered within a2a.server.task_timeout_s fails and is cleaned up.",
)

SUBAGENTS = dict(
    name="subagents", w=880, h=452,
    title="Subagents: one tool, three kinds",
    desc=("The main agent gets one tool, delegate(agent, task). A subagent is another OVAT workflow in this process, "
          "a model behind an OpenAI-compatible API, or an agent reached over A2A. Answers come back framed as information. "
          "Progress is reported as events and shown in the /agents table."),
    header="subagents: [researcher, team_docs, local_api] in workflow.yml",
    cards=[
        card("main", 1, 20, 120, 190, 150, "Main agent", ["one tool:", "delegate(agent, task)", "or you type /ask", "or @name", "shares stop and approver"]),
        card("wf", 2, 420, 60, 440, 92, "workflow",
             ["another OVAT workflow, in this process, with a fresh", "context; it cannot delegate again (one level deep)"]),
        card("api", 3, 420, 176, 440, 92, "api",
             ["any OpenAI-compatible API: a cloud model, a remote OVMS;", "asks first, shows where the task goes; enable_thinking: false"], "tool"),
        card("a2a", 4, 420, 292, 440, 92, "a2a",
             ["another agent over A2A, in another terminal or machine;", "asks first unless it is on this PC; framed as information"]),
        card("table", 5, 20, 316, 190, 80, "/agents", ["a live table of calls;", "x cancels one child"], "ok"),
    ],
    links=[
        link("lw", "M210 170 L418 106"), link("la", "M210 195 L418 222", "tool"),
        link("lb", "M210 220 L418 338"), link("lt", "M115 270 L115 314", "ok"),
    ],
    labels=[],
    steps=[
        step(["main", "wf"], "lw", "A workflow names its subagents. The main agent gets one tool, delegate(agent, task), and the agent argument is the list of those names."),
        step(["main", "api"], "la", "api: an approval panel names the address the task would be sent to. The subagent sees only the task text, not your chat.", "tool"),
        step(["main", "a2a"], "lb", "a2a: the task goes to another agent, in another terminal or on another machine. The token is sent only to the origin you configured."),
        step(["main"], None, "Each answer comes back framed ([Answer from subagent ...] or [Reply from remote agent ...]) so the main model treats it as information, not as instructions.", "ok"),
        step(["main", "table"], "lt", "Progress is reported as subagent_status events. The TUI draws them as the /agents table, and x cancels one child while the parent carries on.", "ok"),
        step(["main"], None, "A subagent shares the parent's stop flag and approver, so Esc stops the children, and an approval a child needs reaches the same panel.", "ok"),
    ],
    foot="Enable a workflow, api or a2a subagent in the subagents: list; see examples/agent/ for a workflow with all three.",
    note="A child that reports no token counts shows a dash, not 0.",
)

DIAGRAMS = [AGENT_TUI, APPROVAL, A2A, SUBAGENTS]


# --------------------------------------------------------------------------
# Standalone animated SVG
# --------------------------------------------------------------------------

def esc(text):
    return html.escape(text, quote=False)


def check_fit(spec):
    """A line must fit its card: ~6.4 px a character at 12.5 px, ~7.4 for mono."""
    for c in spec["cards"]:
        for line in c["lines"]:
            mono = line.startswith("~")
            width = len(line.lstrip("~")) * (7.0 if mono else 6.4)
            assert width <= c["w"] - 26, f"{spec['name']}: '{line}' is too wide for {c['id']} ({width:.0f} > {c['w'] - 26})"
        assert len(c["title"]) * 8.6 <= c["w"] - 26, f"title too wide: {c['title']}"
        needed = 46 + 17 * len(c["lines"]) + 8
        assert needed <= c["h"] + 14, f"{spec['name']}: card {c['id']} is too short for its lines ({needed} > {c['h']})"


def timeline(n, i):
    a, b = i * 100.0 / n, (i + 1) * 100.0 / n
    return a, b


def build_svg(spec, theme):
    check_fit(spec)
    n, secs = len(spec["steps"]), len(spec["steps"]) * 2.4
    dur = f"{secs:g}s"
    by_id = {c["id"]: c for c in spec["cards"]}
    links = {k["id"]: k for k in spec["links"]}

    css = [f"svg{{{LIGHT if theme == 'light' else DARK}}}", BASE_CSS]
    for i in range(n):
        a, b = timeline(n, i)
        r = min(2.0, (b - a) * 0.14)
        stops = [f"0.00%{{opacity:0}}"]
        if a > 0:
            stops.append(f"{a:.2f}%{{opacity:0}}")
        stops.append(f"{a + r:.2f}%{{opacity:1}}")
        stops.append(f"{b - r:.2f}%{{opacity:1}}")
        if b < 100:
            stops.append(f"{b:.2f}%{{opacity:0}}")
            stops.append("100.00%{opacity:0}")
        else:
            stops.append("100.00%{opacity:0}")
        css.append(f"@keyframes s{i}{{{' '.join(stops)}}}.s{i}{{animation:s{i} {dur} linear infinite}}")

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {spec["w"]} {spec["h"]}" width="{spec["w"]}" height="{spec["h"]}" role="img" aria-labelledby="t d">'
           f'<title id="t">{esc(spec["title"])}</title><desc id="d">{esc(spec["desc"])}</desc>',
           "<style>\n" + "\n".join(css) + "\n</style>",
           '<defs><linearGradient id="brand" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0068B5"/>'
           '<stop offset="0.55" stop-color="#00C7FD"/><stop offset="1" stop-color="#8F5CFF"/></linearGradient>'
           '<marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           '<path d="M0 0L10 5L0 10z" class="arrowhead"/></marker></defs>']

    # highlights, one rect per card per step that names it
    for i, s in enumerate(spec["steps"]):
        for cid in s["cards"]:
            c = by_id[cid]
            out.append(f'<rect x="{c["x"] - 1.5}" y="{c["y"] - 1.5}" width="{c["w"] + 3}" height="{c["h"] + 3}" rx="13.5" class="hl s{i}" opacity="0"/>')

    if spec.get("header"):
        out.append(f'<text x="20" y="30" class="mono" style="font-size:13px">{esc(spec["header"])}</text>')
    for k in spec["links"]:
        out.append(f'<path d="{k["d"]}" class="{LINK_CLASS[k["tone"]]}" marker-end="url(#arrow)"/>')
    for (x, y, text, cls, anchor) in spec["labels"]:
        out.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{esc(text)}</text>')

    # travelling dots
    for i, s in enumerate(spec["steps"]):
        if not s["link"]:
            continue
        a, b = timeline(n, i)
        t0, t1 = (a + 0.08 * (b - a)) / 100, (a + 0.82 * (b - a)) / 100
        e = 0.004
        var = TONE_VAR[s["tone"]]
        out.append(
            f'<g class="fx" opacity="0"><circle r="12" fill="var({var})" opacity="0.22"/><circle r="6" fill="var({var})"/>'
            f'<circle r="2.4" fill="#fff" opacity="0.9"/>'
            f'<animateMotion dur="{dur}" repeatCount="indefinite" calcMode="linear" keyTimes="0;{t0:.4f};{t1:.4f};1" keyPoints="0;0;1;1" path="{links[s["link"]]["d"]}"/>'
            f'<animate attributeName="opacity" dur="{dur}" repeatCount="indefinite" keyTimes="0;{t0:.4f};{t0 + e:.4f};{t1:.4f};{t1 + e:.4f};1" values="0;0;1;1;0;0"/></g>')

    # cards on top
    for c in spec["cards"]:
        tone = f" {c['tone']}" if c["tone"] else ""
        out.append(f'<g class="card{tone}"><rect x="{c["x"]}" y="{c["y"]}" width="{c["w"]}" height="{c["h"]}" rx="12" class="box"/>'
                   f'<circle cx="{c["x"] + 20}" cy="{c["y"]}" r="10.5" fill="url(#brand)"/>'
                   f'<text x="{c["x"] + 20}" y="{c["y"] + 4.2}" class="num">{c["num"]}</text>'
                   f'<text x="{c["x"] + 14}" y="{c["y"] + 28}" class="title" style="font-size:14.5px">{esc(c["title"])}</text>')
        for j, line in enumerate(c["lines"]):
            cls = "mono" if line.startswith("~") else "sub"
            out.append(f'<text x="{c["x"] + 14}" y="{c["y"] + 50 + 17 * j}" class="{cls}">{esc(line.lstrip("~"))}</text>')
        out.append("</g>")
    if spec.get("foot"):
        out.append(f'<text x="20" y="{spec["h"] - 14}" class="small">{esc(spec["foot"])}</text>')
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------
# The docs site version: figure[data-seq] with real step text
# --------------------------------------------------------------------------

CONTROLS = ('<div class="diagram-controls">\n'
            '            <button class="ctl" type="button" data-act="toggle" aria-pressed="true"><svg data-icon="pause" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><rect x="3" y="2" width="3.5" height="12" rx="1"/><rect x="9.5" y="2" width="3.5" height="12" rx="1"/></svg><svg data-icon="play" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" hidden><path d="M4 2.5v11l9-5.5z"/></svg><span class="lbl">Pause</span></button>\n'
            '            <button class="ctl" type="button" data-act="prev" aria-label="Previous step"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M10 3L5 8l5 5"/></svg></button>\n'
            '            <button class="ctl" type="button" data-act="next" aria-label="Next step"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M6 3l5 5-5 5"/></svg></button>\n'
            '            <span class="count" aria-live="polite">1 / {n}</span>\n'
            '          </div>')


def build_site(spec):
    name = spec["name"]
    p = name.replace("-", "")[:4]            # short id prefix for path and node ids
    by_id = {c["id"]: c for c in spec["cards"]}
    out = [f'<figure class="diagram" data-seq aria-labelledby="{name}-cap">',
           '        <div class="diagram-stage">',
           f'          <svg class="dg" viewBox="0 0 {spec["w"]} {spec["h"]}" role="img" aria-labelledby="{name}-t {name}-d">',
           f'            <title id="{name}-t">{esc(spec["title"])}</title>',
           f'            <desc id="{name}-d">{esc(spec["desc"])}</desc>',
           f'            <defs><marker id="ah-{name}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0 0 L10 5 L0 10 z"/></marker></defs>']
    for k in spec["links"]:
        out.append(f'            <path id="{p}-{k["id"]}" class="wire" d="{k["d"]}" marker-end="url(#ah-{name})"/>')
    for (x, y, text, cls, anchor) in spec["labels"]:
        out.append(f'            <text class="t-c" x="{x}" y="{y}" text-anchor="{anchor}">{esc(text)}</text>')
    for c in spec["cards"]:
        tone = f" tone-{c['tone']}" if c["tone"] else ""
        g = [f'            <g id="{p}-{c["id"]}" class="node{tone}"><rect class="box" x="{c["x"]}" y="{c["y"]}" width="{c["w"]}" height="{c["h"]}"/>',
             f'<text class="t-k" x="{c["x"] + 16}" y="{c["y"] + 26}">{c["num"]}</text>',
             f'<text class="t-t" x="{c["x"] + 16}" y="{c["y"] + 52}">{esc(c["title"])}</text>']
        for j, line in enumerate(c["lines"]):
            g.append(f'<text class="t-s" x="{c["x"] + 16}" y="{c["y"] + 76 + 19 * j}">{esc(line.lstrip("~"))}</text>')
        g.append("</g>")
        out.append("".join(g))
    out += ["          </svg>", "        </div>",
            f'        <figcaption class="diagram-side" id="{name}-cap">',
            "          " + CONTROLS.replace("{n}", str(len(spec["steps"]))),
            '          <ol class="steps">']
    for s in spec["steps"]:
        attrs = [f'data-nodes="{" ".join(p + "-" + c for c in s["cards"])}"']
        if s["link"]:
            attrs.append(f'data-path="{p}-{s["link"]}"')
        if s["tone"]:
            attrs.append(f'data-tone="{s["tone"]}"')
        out.append(f'            <li {" ".join(attrs)}>{esc(s["text"])}</li>')
    out += ["          </ol>", f'          <p class="diagram-note">{esc(spec["note"])}</p>', "        </figcaption>", "      </figure>"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--site-out", type=Path, help="write the site figures here, one HTML file per diagram folder")
    args = ap.parse_args()
    for spec in DIAGRAMS:
        for theme in ("light", "dark"):
            path = args.out / f"{spec['name']}-{theme}.svg"
            path.write_text(build_svg(spec, theme), encoding="utf-8")
            print("wrote", path.name)
        if args.site_out:
            args.site_out.mkdir(parents=True, exist_ok=True)
            (args.site_out / f"{spec['name']}.html").write_text(build_site(spec) + "\n", encoding="utf-8")
            print("wrote site", spec["name"])


if __name__ == "__main__":
    main()
