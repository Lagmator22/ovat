#!/usr/bin/env python3
"""Draw OVAT's animated Nocturne SVG diagrams for README, docs, and site.

One spec per diagram makes three artifacts, so they cannot drift apart:
  docs/assets/diagrams/<name>-light.svg and -dark.svg   animated, for GitHub
  a <figure data-seq> block for the docs site (ovat-navigate)

Usage:
    python scripts/gen_agent_diagrams.py                 # write all 42 SVGs
    python scripts/gen_agent_diagrams.py --site-out F    # also write site HTML

Every fact in a diagram is verified against the code or AGENTS.md. If a name
stops being accurate, update its spec here and run this script again.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "diagrams"

THEMES = {
    "dark": {
        "bg": "#161826", "bg2": "#1f2236", "frame": "#292b31", "dot": "#262836",
        "surface": "#232532", "border": "#3f424d", "edge": "#75798c",
        "text": "#e9e9ed", "sub": "#b2b6ca", "muted": "#9397ab",
        "accent": "#9184d9", "accentText": "#d2cefd", "trail": "#5d5294",
        "tint": "#2b2741", "group": "#1b1d2b",
    },
    "light": {
        "bg": "#f3f5fe", "bg2": "#f9faff", "frame": "#cfd3e5", "dot": "#dde0ee",
        "surface": "#f9faff", "border": "#b2b6ca", "edge": "#75798c",
        "text": "#292b31", "sub": "#595d6c", "muted": "#75798c",
        "accent": "#796cbf", "accentText": "#5d5294", "trail": "#b5abfc",
        "tint": "#e7e5fe", "group": "#e9ecf8",
    },
}

SANS = "Inter, 'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', 'Cascadia Mono', Menlo, Consolas, 'Liberation Mono', monospace"


def esc(text: object) -> str:
    """Escape text for SVG while preserving double spaces as non-breaking."""
    return (
        str(text)
        .replace("  ", "\u00a0\u00a0")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _num(n: float | int) -> str:
    """Format a number the same way JavaScript String(number) does."""
    if isinstance(n, bool):
        return str(n)
    val = float(n)
    if val == 0.0:
        return "0"
    if val.is_integer():
        return str(int(val))
    return repr(val)


def f(n: float | int) -> str:
    """Round half towards +infinity to two decimals, matching JS Math.round(n*100)/100."""
    val = math.floor(float(n) * 100.0 + 0.5) / 100.0
    return _num(val)


def port(node: dict, side: str, t: float | None = None) -> dict:
    """Return anchor coordinates and orientation on one side of a node."""
    if t is None:
        t = 0.5
    if side == "r":
        return {"x": node["x"] + node["w"], "y": node["y"] + node["h"] * t, "d": "h"}
    if side == "l":
        return {"x": node["x"], "y": node["y"] + node["h"] * t, "d": "h"}
    if side == "t":
        return {"x": node["x"] + node["w"] * t, "y": node["y"], "d": "v"}
    return {"x": node["x"] + node["w"] * t, "y": node["y"] + node["h"], "d": "v"}


def route(edge: dict, nodes_by_id: dict[str, dict]) -> list[dict]:
    """Compute orthogonal polyline control points for an edge."""
    if "pts" in edge:
        return [{"x": p[0], "y": p[1]} for p in edge["pts"]]
    src = nodes_by_id[edge["from"]]
    dst = nodes_by_id[edge["to"]]
    p_start = port(src, edge.get("fs") or "r", edge.get("ft"))
    q_end = port(dst, edge.get("ts") or "l", edge.get("tt"))
    pts = [p_start]
    for v in edge.get("via") or []:
        pts.append({"x": v[0], "y": v[1]})
    if not edge.get("via"):
        if p_start["d"] == "h" and q_end["d"] == "h":
            if abs(p_start["y"] - q_end["y"]) > 0.5:
                mx = edge["mx"] if edge.get("mx") is not None else (p_start["x"] + q_end["x"]) / 2
                pts.extend([{"x": mx, "y": p_start["y"]}, {"x": mx, "y": q_end["y"]}])
        elif p_start["d"] == "v" and q_end["d"] == "v":
            if abs(p_start["x"] - q_end["x"]) > 0.5:
                my = edge["my"] if edge.get("my") is not None else (p_start["y"] + q_end["y"]) / 2
                pts.extend([{"x": p_start["x"], "y": my}, {"x": q_end["x"], "y": my}])
        elif p_start["d"] == "h" and q_end["d"] == "v":
            pts.append({"x": q_end["x"], "y": p_start["y"]})
        else:
            pts.append({"x": p_start["x"], "y": q_end["y"]})
    pts.append(q_end)
    return pts


def path_d(pts: list[dict], radius: float | None = None) -> str:
    """Convert polyline control points into a rounded SVG path string."""
    if radius is None:
        radius = 10
    d = f"M{f(pts[0]['x'])} {f(pts[0]['y'])}"
    for i in range(1, len(pts)):
        pt = pts[i]
        if i < len(pts) - 1:
            prev_pt, next_pt = pts[i - 1], pts[i + 1]
            l1 = math.hypot(pt["x"] - prev_pt["x"], pt["y"] - prev_pt["y"])
            l2 = math.hypot(next_pt["x"] - pt["x"], next_pt["y"] - pt["y"])
            rr = min(radius, l1 / 2, l2 / 2)
            if rr < 1:
                d += f" L{f(pt['x'])} {f(pt['y'])}"
                continue
            p1x = pt["x"] - (pt["x"] - prev_pt["x"]) / l1 * rr
            p1y = pt["y"] - (pt["y"] - prev_pt["y"]) / l1 * rr
            p2x = pt["x"] + (next_pt["x"] - pt["x"]) / l2 * rr
            p2y = pt["y"] + (next_pt["y"] - pt["y"]) / l2 * rr
            d += f" L{f(p1x)} {f(p1y)} Q{f(pt['x'])} {f(pt['y'])} {f(p2x)} {f(p2y)}"
        else:
            d += f" L{f(pt['x'])} {f(pt['y'])}"
    return d


def plen(pts: list[dict]) -> float:
    """Total geometric length of a polyline."""
    return sum(
        math.hypot(pts[i]["x"] - pts[i - 1]["x"], pts[i]["y"] - pts[i - 1]["y"])
        for i in range(1, len(pts))
    )


def check_fit(spec: dict) -> None:
    """Verify node titles and subtitle lines fit cleanly inside each card."""
    for n in spec.get("nodes", []):
        title = n["title"][0] if isinstance(n["title"], list) else n["title"]
        tsz = n.get("size") or (13 if n.get("mono") else 14)
        pad = 20 if (n.get("center") or n.get("pill")) else 24
        t_width = len(title) * (0.61 * tsz if n.get("mono") else 0.54 * tsz)
        assert t_width <= n["w"] - pad, (
            f"{spec['id']}: title '{title}' is too wide for {n['id']} ({t_width:.0f} > {n['w'] - pad})"
        )
        subs = n.get("sub") or []
        if not isinstance(subs, list):
            subs = [subs]
        for s in subs:
            is_code = isinstance(s, dict)
            text = s["c"] if is_code else s
            width = len(text) * (6.9 if is_code else 6.1)
            assert width <= n["w"] - pad, (
                f"{spec['id']}: '{text}' is too wide for {n['id']} ({width:.0f} > {n['w'] - pad})"
            )
        ch = (15 if n.get("k") else 0) + tsz * 0.75 + (7 + len(subs) * 16 - 4 if subs else 0)
        assert ch <= n["h"], (
            f"{spec['id']}: card {n['id']} is too short for its lines ({ch:.1f} > {n['h']})"
        )


def build_svg(spec: dict, theme: str) -> str:
    """Render one animated Nocturne SVG for the given theme ('dark' or 'light')."""
    check_fit(spec)
    c = THEMES[theme]
    w, h = spec["w"], spec["h"]
    sid = re.sub(r"[^a-z0-9]", "", spec["id"], flags=re.I)
    nodes_by_id = {n["id"]: n for n in spec.get("nodes") or []}
    edges_by_id = {e["id"]: e for e in spec.get("edges") or []}
    steps = spec.get("steps") or []
    fade, tr = 0.3, 0.75
    hold = spec["hold"] if spec.get("hold") is not None else 1.6

    t = 0.0
    step_starts: list[float] = []
    for s in steps:
        step_starts.append(t)
        t += s.get("d") or 3
    total = t + hold

    def pct(x: float) -> str:
        clamped = max(0.0, min(total, x))
        return f"{clamped / total * 100:.3f}%"

    css: list[str] = []
    kf = 0

    def anim(cls: str, frames: str) -> None:
        nonlocal kf
        nm = f"k{kf}"
        kf += 1
        css.append(f"@keyframes {nm}{{{frames}}}.{cls}{{animation:{nm} {f(total)}s linear infinite}}")

    def op_frames(wins: list[tuple[float, float]]) -> str:
        out = ["0%{opacity:0}"]
        for a, b in wins:
            out.extend([
                f"{pct(a)}{{opacity:0}}",
                f"{pct(a + fade)}{{opacity:1}}",
                f"{pct(b - fade)}{{opacity:1}}",
                f"{pct(b)}{{opacity:0}}",
            ])
        out.append("100%{opacity:0}")
        return "".join(out)

    act: dict[str, list[tuple[float, float]]] = {}
    first: dict[str, float] = {}
    edge_occ: list[tuple[str, float]] = []
    for i, s in enumerate(steps):
        s_start = step_starts[i]
        end = total - fade if i == len(steps) - 1 else s_start + (s.get("d") or 3)
        st = s_start + (tr * 0.8 if s.get("e") else 0.0)
        for nid in s.get("n") or []:
            act.setdefault(nid, []).append((st, end))
            if nid not in first:
                first[nid] = st
        for eid in s.get("e") or []:
            edge_occ.append((eid, s_start))

    o: list[str] = []
    o.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img" aria-labelledby="t{sid} d{sid}">'
    )
    o.append(f'<title id="t{sid}">{esc(spec["title"])}</title><desc id="d{sid}">{esc(spec["desc"])}</desc>')
    o.append(
        "<defs>"
        f'<radialGradient id="bg{sid}" cx="0.12" cy="0" r="1.1">'
        f'<stop offset="0" stop-color="{c["bg2"]}"/><stop offset="0.7" stop-color="{c["bg"]}"/></radialGradient>'
        f'<linearGradient id="rule{sid}" x1="0" x2="1">'
        f'<stop offset="0" stop-color="{c["border"]}" stop-opacity="0"/>'
        f'<stop offset="0.08" stop-color="{c["border"]}"/>'
        f'<stop offset="0.92" stop-color="{c["border"]}"/>'
        f'<stop offset="1" stop-color="{c["border"]}" stop-opacity="0"/></linearGradient>'
        f'<pattern id="dp{sid}" width="16" height="16" patternUnits="userSpaceOnUse">'
        f'<circle cx="1" cy="1" r="0.9" fill="{c["dot"]}"/></pattern>'
        f'<marker id="ah{sid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M1 1.5L8.5 5L1 8.5" fill="none" stroke="{c["edge"]}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></marker>'
        "</defs>"
    )
    o.append(f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="8" fill="url(#bg{sid})" stroke="{c["frame"]}"/>')
    o.append(f'<rect x="12" y="48" width="{w - 24}" height="{h - 108}" fill="url(#dp{sid})"/>')
    o.append(f'<text x="28" y="34" style="font:500 11px {MONO};letter-spacing:.14em" fill="{c["accent"]}">OVAT</text>')
    o.append(f'<text x="72" y="34" style="font:500 11px {MONO};letter-spacing:.14em" fill="{c["sub"]}">{esc(spec["kicker"].upper())}</text>')
    if spec.get("cmd"):
        o.append(f'<text x="{w - 28}" y="34" text-anchor="end" style="font:400 11.5px {MONO}" fill="{c["muted"]}">{esc(spec["cmd"])}</text>')

    for g in spec.get("groups") or []:
        o.append(
            f'<rect x="{_num(g["x"])}" y="{_num(g["y"])}" width="{_num(g["w"])}" height="{_num(g["h"])}" '
            f'rx="8" fill="{c["group"]}" stroke="{c["border"]}" stroke-width="0.8" stroke-dasharray="1 3"/>'
        )
        if g.get("label"):
            o.append(
                f'<text x="{_num(g["x"] + 12)}" y="{_num(g["y"] + 18)}" '
                f'style="font:500 10.5px {MONO};letter-spacing:.12em" fill="{c["muted"]}">{esc(g["label"].upper())}</text>'
            )
        if g.get("text"):
            sub_span = (
                f'<tspan dx="12" style="font-weight:400;font-size:12.5px" fill="{c["sub"]}">{esc(g["sub"])}</tspan>'
                if g.get("sub") else ""
            )
            o.append(
                f'<text x="{_num(g["x"] + 16)}" y="{_num(g["y"] + g["h"] / 2 + 5)}" '
                f'style="font:500 13.5px {SANS}" fill="{c["text"]}">{esc(g["text"])}{sub_span}</text>'
            )

    for ln in spec.get("lines") or []:
        dash = ln.get("dash") or "2 5"
        o.append(f'<path d="M{_num(ln["x1"])} {_num(ln["y1"])} L{_num(ln["x2"])} {_num(ln["y2"])}" stroke="{c["border"]}" stroke-width="1" stroke-dasharray="{dash}" fill="none"/>')

    for x in spec.get("texts") or []:
        st = (
            f'font:400 {x.get("size") or 11.5}px {MONO}'
            if x.get("mono")
            else f'font:{x.get("weight") or 400} {x.get("size") or 12.5}px {SANS}'
        )
        ls = f';letter-spacing:{x["ls"]}' if x.get("ls") else ""
        anc = f' text-anchor="{x["anchor"]}"' if x.get("anchor") else ""
        o.append(f'<text x="{_num(x["x"])}" y="{_num(x["y"])}"{anc} style="{st}{ls}" fill="{c[x.get("color") or "sub"]}">{esc(x["s"])}</text>')

    routed: dict[str, list[dict]] = {}
    for e in spec.get("edges") or []:
        pts = route(e, nodes_by_id)
        routed[e["id"]] = pts
        d = path_d(pts, e.get("r"))
        dash_attr = ' stroke-dasharray="3 4"' if e.get("dashed") else ""
        arrow_attr = "" if e.get("noArrow") else f' marker-end="url(#ah{sid})"'
        o.append(f'<path d="{d}" fill="none" stroke="{c["edge"]}" stroke-width="1.3"{dash_attr}{arrow_attr}/>')

    for i, (eid, occ_s) in enumerate(edge_occ):
        pts = routed.get(eid)
        if not pts:
            continue
        d = path_d(pts, edges_by_id[eid].get("r"))
        length = plen(pts) + 4
        seg = min(46.0, length * 0.6)
        tc, cc = f"tr{i}", f"cm{i}"
        o.append(
            f'<path class="tr {tc}" d="{d}" fill="none" stroke="{c["trail"]}" '
            f'stroke-width="1.8" stroke-dasharray="{f(length)} {f(length)}" stroke-dashoffset="{f(length)}"/>'
        )
        anim(
            tc,
            f"0%{{stroke-dashoffset:{f(length)};opacity:1}}"
            f"{pct(occ_s)}{{stroke-dashoffset:{f(length)};opacity:1}}"
            f"{pct(occ_s + tr)}{{stroke-dashoffset:0}}"
            f"{pct(total - fade)}{{stroke-dashoffset:0;opacity:1}}"
            f"100%{{stroke-dashoffset:0;opacity:0}}",
        )
        o.append(
            f'<path class="cmt {cc}" d="{d}" fill="none" stroke="{c["accent"]}" '
            f'stroke-width="2.6" stroke-linecap="round" stroke-dasharray="{f(seg)} {f(length + seg + 20)}" stroke-dashoffset="{f(seg)}" opacity="0"/>'
        )
        anim(
            cc,
            f"0%{{stroke-dashoffset:{f(seg)};opacity:0}}"
            f"{pct(occ_s)}{{stroke-dashoffset:{f(seg)};opacity:0}}"
            f"{pct(occ_s + 0.02)}{{opacity:1}}"
            f"{pct(occ_s + tr)}{{stroke-dashoffset:{f(-length)};opacity:1}}"
            f"{pct(occ_s + tr + 0.05)}{{stroke-dashoffset:{f(-length)};opacity:0}}"
            f"100%{{stroke-dashoffset:{f(-length)};opacity:0}}",
        )

    for e in spec.get("edges") or []:
        if not e.get("label"):
            continue
        pts = routed[e["id"]]
        if e.get("lx") is not None:
            lx, ly = e["lx"], e["ly"]
        else:
            bi, bl = 1, 0.0
            for i in range(1, len(pts)):
                seg_len = math.hypot(pts[i]["x"] - pts[i - 1]["x"], pts[i]["y"] - pts[i - 1]["y"])
                if seg_len > bl:
                    bl = seg_len
                    bi = i
            lx = (pts[bi]["x"] + pts[bi - 1]["x"]) / 2
            ly = (pts[bi]["y"] + pts[bi - 1]["y"]) / 2
        lines = e["label"] if isinstance(e["label"], list) else [e["label"]]
        wd = max(len(line) for line in lines) * 6.7 + 14
        ht = len(lines) * 14 + 8
        o.append(
            f'<rect x="{f(lx - wd / 2)}" y="{f(ly - ht / 2)}" width="{f(wd)}" height="{_num(ht)}" '
            f'rx="{_num(min(ht / 2, 9))}" fill="{c["bg"]}" stroke="{c["border"]}" stroke-width="0.8"/>'
        )
        for i, line in enumerate(lines):
            o.append(
                f'<text x="{f(lx)}" y="{f(ly - ht / 2 + 4 + 11 + i * 14)}" '
                f'text-anchor="middle" style="font:400 11px {MONO}" fill="{c["sub"]}">{esc(line)}</text>'
            )

    gly = {"q": "?", "ok": "\u2713", "fail": "\u2715", "warn": "!"}
    for n in spec.get("nodes") or []:
        rx = n["h"] / 2 if n.get("pill") else 6
        kind = n.get("kind")
        dash = kind in ("fail", "warn") or n.get("optional")
        clean_nid = re.sub(r"[^a-z0-9]", "", n["id"], flags=re.I)
        vc, ac = f"v{clean_nid}", f"a{clean_nid}"
        o.append("<g>")
        fill_col = c["group"] if kind == "q" else c["surface"]
        dash_attr = ' stroke-dasharray="4 3"' if dash else ""
        o.append(
            f'<rect x="{_num(n["x"])}" y="{_num(n["y"])}" width="{_num(n["w"])}" height="{_num(n["h"])}" '
            f'rx="{_num(rx)}" fill="{fill_col}" stroke="{c["border"]}" stroke-width="1"{dash_attr}/>'
        )
        if n["id"] in act:
            o.append(
                f'<rect class="v {vc}" x="{_num(n["x"])}" y="{_num(n["y"])}" width="{_num(n["w"])}" height="{_num(n["h"])}" '
                f'rx="{_num(rx)}" fill="{c["tint"]}" stroke="{c["trail"]}" stroke-width="1.2" opacity="0"/>'
            )
            anim(
                vc,
                f"0%{{opacity:0}}{pct(first[n['id']])}{{opacity:0}}"
                f"{pct(first[n['id']] + fade)}{{opacity:1}}"
                f"{pct(total - fade)}{{opacity:1}}100%{{opacity:0}}",
            )
            corner = (
                ""
                if n.get("pill")
                else (
                    f'<path d="M{_num(n["x"])} {_num(n["y"] + 14)} V{_num(n["y"] + 6)} '
                    f'Q{_num(n["x"])} {_num(n["y"])} {_num(n["x"] + 6)} {_num(n["y"])} H{_num(n["x"] + 16)}" '
                    f'fill="none" stroke="{c["accent"]}" stroke-width="3.5" stroke-linecap="square"/>'
                )
            )
            o.append(
                f'<g class="{ac}" opacity="0">'
                f'<rect x="{_num(n["x"])}" y="{_num(n["y"])}" width="{_num(n["w"])}" height="{_num(n["h"])}" '
                f'rx="{_num(rx)}" fill="none" stroke="{c["accent"]}" stroke-width="1.8"/>{corner}</g>'
            )
            anim(ac, op_frames(act[n["id"]]))

        subs = n.get("sub") if n.get("sub") else []
        if not isinstance(subs, list):
            subs = [subs]
        k = n.get("k")
        mono = n.get("mono")
        center = n.get("center") or n.get("pill")
        tsz = n.get("size") or (13 if mono else 14)
        ch = (15 if k else 0) + tsz * 0.75 + (7 + len(subs) * 16 - 4 if subs else 0)
        y0 = n["y"] + (n["h"] - ch) / 2
        tx = n["x"] + n["w"] / 2 if center else n["x"] + 14
        anc = ' text-anchor="middle"' if center else ""
        if k:
            o.append(
                f'<text x="{_num(tx)}" y="{f(y0 + 8)}"{anc} '
                f'style="font:500 10px {MONO};letter-spacing:.12em" fill="{c["muted"]}">{esc(k.upper())}</text>'
            )
        tb = y0 + (15 if k else 0) + tsz * 0.75
        tl = n["title"] if isinstance(n["title"], list) else [n["title"]]
        title_font = f"500 {_num(tsz)}px {MONO}" if mono else f"500 {_num(tsz)}px {SANS}"
        title_ls = "0" if mono else "-.01em"
        title_fill = c["accentText"] if kind == "ok" else c["text"]
        o.append(
            f'<text x="{_num(tx)}" y="{f(tb)}"{anc} '
            f'style="font:{title_font};letter-spacing:{title_ls}" fill="{title_fill}">{esc(tl[0])}</text>'
        )
        for i, s in enumerate(subs):
            is_code = isinstance(s, dict)
            str_s = s["c"] if is_code else s
            sub_font = f"11.5px {MONO}" if is_code else f"12.5px {SANS}"
            sub_fill = c["accentText"] if is_code else c["sub"]
            o.append(
                f'<text x="{_num(tx)}" y="{f(tb + 7 + 12 + i * 16)}"{anc} '
                f'style="font:400 {sub_font}" fill="{sub_fill}">{esc(str_s)}</text>'
            )
        if kind in gly and not n.get("pill"):
            gx, gy = n["x"] + n["w"] - 16, n["y"] + 16
            circle_stroke = c["accent"] if kind == "ok" else c["muted"]
            o.append(f'<circle cx="{_num(gx)}" cy="{_num(gy)}" r="8" fill="none" stroke="{circle_stroke}" stroke-width="1"/>')
            if kind == "ok":
                o.append(
                    f'<path d="M{_num(gx - 3.5)} {_num(gy)} L{_num(gx - 1)} {_num(gy + 2.6)} L{_num(gx + 3.6)} {_num(gy - 2.6)}" '
                    f'fill="none" stroke="{c["accent"]}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
                )
            elif kind == "fail":
                o.append(
                    f'<path d="M{_num(gx - 2.8)} {_num(gy - 2.8)} L{_num(gx + 2.8)} {_num(gy + 2.8)} '
                    f'M{_num(gx + 2.8)} {_num(gy - 2.8)} L{_num(gx - 2.8)} {_num(gy + 2.8)}" '
                    f'fill="none" stroke="{c["sub"]}" stroke-width="1.4" stroke-linecap="round"/>'
                )
            else:
                o.append(
                    f'<text x="{_num(gx)}" y="{_num(gy + 3.8)}" text-anchor="middle" '
                    f'style="font:600 10.5px {MONO}" fill="{c["sub"]}">{gly[kind]}</text>'
                )
        o.append("</g>")

    cy = h - 26
    o.append(f'<rect x="28" y="{h - 54}" width="{w - 56}" height="1" fill="url(#rule{sid})"/>')
    tw, tg = 12, 4
    tx0 = w - 28 - len(steps) * (tw + tg) + tg
    for i, s in enumerate(steps):
        x = tx0 + i * (tw + tg)
        o.append(f'<rect x="{x}" y="{cy - 4}" width="{tw}" height="2" rx="1" fill="{c["border"]}"/>')
        tc = f"tk{i}"
        o.append(f'<rect class="{tc}" x="{x}" y="{cy - 4}" width="{tw}" height="2" rx="1" fill="{c["accent"]}" opacity="0"/>')
        end = total if i == len(steps) - 1 else step_starts[i] + (s.get("d") or 3)
        anim(tc, op_frames([(step_starts[i], end)]))
        cc = f"cp{i}"
        nn = f"{i + 1:02d} / {len(steps):02d}"
        o.append(
            f'<g class="cap {cc}" opacity="0">'
            f'<text x="28" y="{cy}" style="font:500 11px {MONO};letter-spacing:.06em" fill="{c["accent"]}">{nn}</text>'
            f'<text x="92" y="{cy}" style="font:400 13.5px {SANS}" fill="{c["text"]}">{esc(s["c"])}</text></g>'
        )
        anim(cc, op_frames([(step_starts[i], end)]))

    o.append(f'<text class="sum" x="28" y="{cy}" style="font:400 13.5px {SANS}" fill="{c["text"]}" opacity="0">{esc(spec.get("summary") or "")}</text>')
    css.append("@media (prefers-reduced-motion: reduce){*{animation:none!important}.v{opacity:1!important}.cap,.cmt{opacity:0!important}.sum{opacity:1!important}.tr{stroke-dashoffset:0!important}}")
    o.insert(3, "<style>" + "\n".join(css) + "</style>")
    o.append("</svg>")
    return "\n".join(o)


CONTROLS = (
    '<div class="diagram-controls">\n'
    '            <button class="ctl" type="button" data-act="toggle" aria-pressed="true"><svg data-icon="pause" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><rect x="3" y="2" width="3.5" height="12" rx="1"/><rect x="9.5" y="2" width="3.5" height="12" rx="1"/></svg><svg data-icon="play" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" hidden><path d="M4 2.5v11l9-5.5z"/></svg><span class="lbl">Pause</span></button>\n'
    '            <button class="ctl" type="button" data-act="prev" aria-label="Previous step"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M10 3L5 8l5 5"/></svg></button>\n'
    '            <button class="ctl" type="button" data-act="next" aria-label="Next step"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M6 3l5 5-5 5"/></svg></button>\n'
    '            <span class="count" aria-live="polite">1 / {n}</span>\n'
    "          </div>"
)


def build_site(spec: dict) -> str:
    """Build the interactive <figure data-seq> HTML block for the docs site."""
    name = spec["id"]
    p = re.sub(r"[^a-z0-9]", "", name, flags=re.I)[:4]
    nodes_by_id = {n["id"]: n for n in spec.get("nodes") or []}
    out = [
        f'<figure class="diagram" data-seq aria-labelledby="{name}-cap">',
        '        <div class="diagram-stage">',
        f'          <svg class="dg" viewBox="0 0 {spec["w"]} {spec["h"]}" role="img" aria-labelledby="{name}-t {name}-d">',
        f'            <title id="{name}-t">{esc(spec["title"])}</title>',
        f'            <desc id="{name}-d">{esc(spec["desc"])}</desc>',
        f'            <defs><marker id="ah-{name}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0 0 L10 5 L0 10 z"/></marker></defs>',
    ]
    for e in spec.get("edges") or []:
        d = path_d(route(e, nodes_by_id), e.get("r"))
        out.append(f'            <path id="{p}-{e["id"]}" class="wire" d="{d}" marker-end="url(#ah-{name})"/>')
    for x in spec.get("texts") or []:
        out.append(f'            <text class="t-c" x="{_num(x["x"])}" y="{_num(x["y"])}">{esc(x["s"])}</text>')
    for n in spec.get("nodes") or []:
        kind = n.get("kind") or ""
        tone = f" tone-{kind}" if kind else ""
        g = [
            f'            <g id="{p}-{n["id"]}" class="node{tone}"><rect class="box" x="{_num(n["x"])}" y="{_num(n["y"])}" width="{_num(n["w"])}" height="{_num(n["h"])}"/>',
            f'<text class="t-k" x="{_num(n["x"] + 14)}" y="{_num(n["y"] + 22)}">{esc(n.get("k") or "")}</text>',
            f'<text class="t-t" x="{_num(n["x"] + 14)}" y="{_num(n["y"] + 42)}">{esc(n["title"][0] if isinstance(n["title"], list) else n["title"])}</text>',
        ]
        subs = n.get("sub") or []
        if not isinstance(subs, list):
            subs = [subs]
        for j, sub in enumerate(subs):
            str_s = sub["c"] if isinstance(sub, dict) else sub
            g.append(f'<text class="t-s" x="{_num(n["x"] + 14)}" y="{_num(n["y"] + 62 + 16 * j)}">{esc(str_s)}</text>')
        g.append("</g>")
        out.append("".join(g))
    out += [
        "          </svg>",
        "        </div>",
        f'        <figcaption class="diagram-side" id="{name}-cap">',
        "          " + CONTROLS.replace("{n}", str(len(spec["steps"]))),
        '          <ol class="steps">',
    ]
    for s in spec["steps"]:
        attrs = [f'data-nodes="{" ".join(p + "-" + nid for nid in (s.get("n") or []))}"']
        if s.get("e"):
            attrs.append(f'data-path="{" ".join(p + "-" + eid for eid in s["e"])}"')
        out.append(f'            <li {" ".join(attrs)}>{esc(s["c"])}</li>')
    note = spec.get("note") or spec.get("summary") or ""
    out += ["          </ol>", f'          <p class="diagram-note">{esc(note)}</p>', "        </figcaption>", "      </figure>"]
    return "\n".join(out)


SPECS_README = [{'id': 'core-flow',
  'w': 880,
  'h': 400,
  'kicker': 'one run, end to end',
  'cmd': '$ ovat run workflow.yml -i "what do my notes say?"',
  'title': 'How one ovat run request flows',
  'desc': 'workflow.yml is checked against a strict schema, the agent factory wires the model, '
          'tools and search, one of four engines runs the loop, and the model runs on an OVMS '
          'server or locally through openvino_genai, on an Intel CPU, GPU or NPU. The answer comes '
          'back with its sources.',
  'summary': 'workflow.yml, check, agent factory, engine, then OVMS or a local model on an Intel '
             'CPU, GPU or NPU. Answer + sources.',
  'groups': [{'x': 694, 'y': 64, 'w': 166, 'h': 204, 'label': 'runs the model'},
             {'x': 24,
              'y': 284,
              'w': 836,
              'h': 50,
              'text': 'Intel hardware',
              'sub': 'model.device picks where the model runs'}],
  'nodes': [{'id': 'yaml',
             'x': 24,
             'y': 84,
             'w': 140,
             'h': 104,
             'k': '01',
             'title': 'workflow.yml',
             'mono': True,
             'sub': ['the one file', 'you write']},
            {'id': 'check',
             'x': 190,
             'y': 84,
             'w': 140,
             'h': 104,
             'k': '02',
             'title': 'Check',
             'sub': ['strict schema,', 'a typo is an error']},
            {'id': 'factory',
             'x': 356,
             'y': 84,
             'w': 140,
             'h': 104,
             'k': '03',
             'title': 'Agent factory',
             'sub': ['wires the model,', 'tools, search', 'and subagents']},
            {'id': 'engine',
             'x': 522,
             'y': 84,
             'w': 140,
             'h': 104,
             'k': '04',
             'title': 'Engine',
             'sub': ['native, react,', 'llamaindex or', 'openai-agents']},
            {'id': 'ovms',
             'x': 706,
             'y': 90,
             'w': 142,
             'h': 76,
             'k': '05',
             'title': 'OVMS server',
             'sub': ['decodes tool calls', 'Windows, Linux']},
            {'id': 'local',
             'x': 706,
             'y': 178,
             'w': 142,
             'h': 76,
             'title': 'Local model',
             'sub': ['openvino_genai', 'no tools, any OS']},
            {'id': 'answer',
             'x': 24,
             'y': 214,
             'w': 140,
             'h': 34,
             'pill': True,
             'kind': 'ok',
             'title': 'answer + sources',
             'size': 13},
            {'id': 'cpu',
             'x': 616,
             'y': 293,
             'w': 72,
             'h': 32,
             'center': True,
             'mono': True,
             'title': 'CPU'},
            {'id': 'gpu',
             'x': 700,
             'y': 293,
             'w': 72,
             'h': 32,
             'center': True,
             'mono': True,
             'title': 'GPU'},
            {'id': 'npu',
             'x': 784,
             'y': 293,
             'w': 72,
             'h': 32,
             'center': True,
             'mono': True,
             'title': 'NPU'}],
  'edges': [{'id': 'e1', 'from': 'yaml', 'to': 'check'},
            {'id': 'e2', 'from': 'check', 'to': 'factory'},
            {'id': 'e3', 'from': 'factory', 'to': 'engine'},
            {'id': 'e4', 'from': 'engine', 'to': 'ovms', 'ft': 0.4},
            {'id': 'e5', 'from': 'engine', 'to': 'local', 'ft': 0.75, 'tt': 0.3},
            {'id': 'e6', 'from': 'local', 'to': 'gpu', 'fs': 'b', 'ts': 't', 'ft': 0.5, 'my': 274},
            {'id': 'e7', 'pts': [[706, 231], [164, 231]]}],
  'steps': [{'n': ['yaml'],
             'c': 'You write one file, workflow.yml: the model, its device, the tools and the '
                  'engine.'},
            {'e': ['e1'],
             'n': ['check'],
             'c': 'OVAT checks it against a strict schema. A misspelt key fails at once, by name.'},
            {'e': ['e2'],
             'n': ['factory'],
             'c': 'The agent factory wires up the model, the tools, document search and '
                  'subagents.'},
            {'e': ['e3'],
             'n': ['engine'],
             'c': 'agent.type picks the engine that runs the loop. Nothing else in the file '
                  'changes.'},
            {'e': ['e4'],
             'n': ['ovms'],
             'c': 'On Windows and Linux the model runs on an OVMS server, which decodes tool '
                  'calls.'},
            {'e': ['e5'],
             'n': ['local'],
             'c': 'For local chat on any OS, openvino_genai runs the model inside OVAT. No tools.'},
            {'e': ['e6'],
             'n': ['cpu', 'gpu', 'npu'],
             'c': 'model.device picks the Intel CPU, GPU or NPU that the model runs on.'},
            {'e': ['e7'],
             'n': ['answer'],
             'c': 'The answer comes back with the files it came from. Nothing left the machine.',
             'd': 3.6}],
  'name': 'core-flow'},
 {'id': 'agent-loop',
  'w': 880,
  'h': 380,
  'kicker': 'the agent loop',
  'title': 'The agent loop',
  'desc': 'The question goes to the model; a reply that asks for a tool makes the tool run as '
          'Python on your machine; the result goes back to the model; a reply with no tool call is '
          'the answer, shown with its sources. The loop stops after max_iterations rounds.',
  'summary': 'Ask the model. If it wants a tool, run it here and add the result. No tool call '
             'means the answer.',
  'nodes': [{'id': 'q',
             'x': 24,
             'y': 110,
             'w': 150,
             'h': 40,
             'pill': True,
             'title': 'your question',
             'size': 13},
            {'id': 'model',
             'x': 236,
             'y': 90,
             'w': 180,
             'h': 80,
             'k': '1 ask',
             'title': 'Model',
             'sub': ['history + tool menu', 'on OVMS']},
            {'id': 'dec',
             'x': 480,
             'y': 96,
             'w': 180,
             'h': 68,
             'kind': 'q',
             'k': '2 read',
             'title': 'Asks for a tool?'},
            {'id': 'tool',
             'x': 480,
             'y': 220,
             'w': 180,
             'h': 80,
             'k': '3 act',
             'title': 'Tool runs',
             'sub': ['plain Python,', 'on your machine']},
            {'id': 'result',
             'x': 236,
             'y': 220,
             'w': 180,
             'h': 80,
             'k': '4 report',
             'title': 'Result',
             'sub': ['added to the', 'history']},
            {'id': 'answer',
             'x': 712,
             'y': 110,
             'w': 148,
             'h': 40,
             'pill': True,
             'kind': 'ok',
             'title': 'answer + sources',
             'size': 13}],
  'texts': [{'x': 24, 'y': 268, 'mono': True, 's': 'max_iterations', 'color': 'muted'},
            {'x': 24, 'y': 286, 's': 'caps the rounds,', 'color': 'muted', 'size': 12},
            {'x': 24, 'y': 302, 's': 'so it always ends', 'color': 'muted', 'size': 12}],
  'edges': [{'id': 'qm', 'from': 'q', 'to': 'model'},
            {'id': 'l1', 'from': 'model', 'to': 'dec'},
            {'id': 'l2', 'from': 'dec', 'to': 'tool', 'fs': 'b', 'ts': 't', 'label': 'yes'},
            {'id': 'l3', 'from': 'tool', 'to': 'result', 'fs': 'l', 'ts': 'r'},
            {'id': 'l4', 'from': 'result', 'to': 'model', 'fs': 't', 'ts': 'b'},
            {'id': 'l5', 'from': 'dec', 'to': 'answer', 'label': 'no', 'lx': 686, 'ly': 130}],
  'steps': [{'n': ['q'], 'c': 'You ask a question.', 'd': 2.2},
            {'e': ['qm'],
             'n': ['model'],
             'c': 'OVAT sends it to the model, with the history and a menu of tools.'},
            {'e': ['l1'], 'n': ['dec'], 'c': 'OVAT reads the reply. Does it ask for a tool?'},
            {'e': ['l2'],
             'n': ['tool'],
             'c': 'Yes: OVAT runs that tool as ordinary Python, on your own machine.'},
            {'e': ['l3'], 'n': ['result'], 'c': "The tool's result is added to the history."},
            {'e': ['l4'],
             'n': ['model'],
             'c': 'The model is asked again, now with the result in front of it.'},
            {'e': ['l1'],
             'n': ['dec'],
             'c': 'Each round can call more tools, until the model has what it needs.'},
            {'e': ['l5'],
             'n': ['answer'],
             'c': 'A reply with no tool call is the answer, printed with its sources.',
             'd': 3.6}],
  'name': 'agent-loop'},
 {'id': 'engines',
  'w': 880,
  'h': 380,
  'kicker': 'one config, four engines',
  'title': 'One config, four agent engines',
  'desc': 'agent.type in workflow.yml selects one of four engines: native (built in), react '
          '(LangChain), llamaindex or openai-agents. All four talk to the same OVMS server with '
          'the same model, tools and endpoint.',
  'summary': 'agent.type picks native, react, llamaindex or openai-agents. Same server, model and '
             'tools for all four.',
  'nodes': [{'id': 'yaml',
             'x': 24,
             'y': 130,
             'w': 186,
             'h': 96,
             'title': 'workflow.yml',
             'mono': True,
             'sub': [{'c': 'agent:'}, {'c': '  type: native'}]},
            {'id': 'native',
             'x': 290,
             'y': 70,
             'w': 230,
             'h': 52,
             'title': 'native',
             'mono': True,
             'sub': ["OVAT's own loop, no extras"]},
            {'id': 'react',
             'x': 290,
             'y': 132,
             'w': 230,
             'h': 52,
             'title': 'react',
             'mono': True,
             'sub': ['LangChain']},
            {'id': 'llama',
             'x': 290,
             'y': 194,
             'w': 230,
             'h': 52,
             'title': 'llamaindex',
             'mono': True,
             'sub': ['LlamaIndex FunctionAgent']},
            {'id': 'oai',
             'x': 290,
             'y': 256,
             'w': 230,
             'h': 52,
             'title': 'openai-agents',
             'mono': True,
             'sub': ['OpenAI Agents SDK']},
            {'id': 'ovms',
             'x': 604,
             'y': 110,
             'w': 256,
             'h': 136,
             'k': 'same for all four',
             'title': 'OVMS  /v3',
             'sub': ['the same server and model',
                     'the same tools',
                     'tokens recorded on every reply']}],
  'edges': [{'id': 'y1', 'from': 'yaml', 'to': 'native', 'mx': 250},
            {'id': 'y2', 'from': 'yaml', 'to': 'react', 'mx': 250},
            {'id': 'y3', 'from': 'yaml', 'to': 'llama', 'mx': 250},
            {'id': 'y4', 'from': 'yaml', 'to': 'oai', 'mx': 250},
            {'id': 'o1', 'from': 'native', 'to': 'ovms', 'tt': 0.2, 'mx': 562},
            {'id': 'o2', 'from': 'react', 'to': 'ovms', 'tt': 0.4, 'mx': 562},
            {'id': 'o3', 'from': 'llama', 'to': 'ovms', 'tt': 0.6, 'mx': 562},
            {'id': 'o4', 'from': 'oai', 'to': 'ovms', 'tt': 0.8, 'mx': 562}],
  'steps': [{'n': ['yaml'],
             'c': 'agent.type is the one line that picks the engine. Tools and prompt stay as they '
                  'are.'},
            {'e': ['y1', 'o1'],
             'n': ['native', 'ovms'],
             'c': "native: OVAT's own loop. It needs nothing extra."},
            {'e': ['y2', 'o2'],
             'n': ['react', 'ovms'],
             'c': 'react: the same agent, run through LangChain.'},
            {'e': ['y3', 'o3'],
             'n': ['llama', 'ovms'],
             'c': 'llamaindex: the same agent, run through LlamaIndex.'},
            {'e': ['y4', 'o4'],
             'n': ['oai', 'ovms'],
             'c': 'openai-agents: the OpenAI Agents SDK, pointed at the local server, not '
                  "OpenAI's."},
            {'e': ['o1', 'o2', 'o3', 'o4'],
             'n': ['ovms'],
             'c': 'All four use the same server and model. ovat bench runs them side by side.',
             'd': 3.6}],
  'name': 'engines'},
 {'id': 'rag',
  'w': 880,
  'h': 400,
  'kicker': 'search your documents',
  'cmd': '$ ovat index ./docs workflow.yml',
  'title': 'RAG in two steps',
  'desc': 'Index once: files are cut into chunks, turned into 384 numbers each by bge-small, and '
          'stored in a sqlite-vec file. Every question: the question is turned into numbers, the '
          'closest chunks are found, and the model answers from them and names the source files.',
  'summary': 'Index once: files to chunks to 384 numbers to sqlite-vec. Each question: embed, find '
             'the closest chunks, answer with sources.',
  'groups': [{'x': 24, 'y': 64, 'w': 836, 'h': 118, 'label': 'index once'},
             {'x': 24, 'y': 198, 'w': 836, 'h': 134, 'label': 'every question'}],
  'nodes': [{'id': 'files',
             'x': 40,
             'y': 92,
             'w': 160,
             'h': 76,
             'title': 'Your files',
             'sub': ['a folder of', '.md and .txt']},
            {'id': 'chunks',
             'x': 236,
             'y': 92,
             'w': 160,
             'h': 76,
             'title': 'Chunks',
             'sub': ['cut into', 'short passages']},
            {'id': 'emb',
             'x': 432,
             'y': 92,
             'w': 160,
             'h': 76,
             'title': 'bge-small',
             'mono': True,
             'sub': ['384 numbers', 'per chunk']},
            {'id': 'store',
             'x': 628,
             'y': 92,
             'w': 216,
             'h': 76,
             'title': 'sqlite-vec file',
             'sub': ['one file on disk, remembers', 'when each file was read']},
            {'id': 'q',
             'x': 40,
             'y': 228,
             'w': 160,
             'h': 88,
             'title': 'Question',
             'sub': ['"what is OVAT\'s', 'memory budget?"']},
            {'id': 'qe',
             'x': 236,
             'y': 228,
             'w': 160,
             'h': 88,
             'title': 'bge-small',
             'mono': True,
             'sub': ['the question as', '384 numbers']},
            {'id': 'near',
             'x': 432,
             'y': 228,
             'w': 160,
             'h': 88,
             'title': 'Closest chunks',
             'sub': ['nearest neighbours', 'in the index']},
            {'id': 'ans',
             'x': 628,
             'y': 228,
             'w': 216,
             'h': 88,
             'kind': 'ok',
             'title': 'Model answers',
             'sub': ['from those passages', {'c': 'sources: notes/q3.md'}]}],
  'edges': [{'id': 'a1', 'from': 'files', 'to': 'chunks'},
            {'id': 'a2', 'from': 'chunks', 'to': 'emb'},
            {'id': 'a3', 'from': 'emb', 'to': 'store'},
            {'id': 'b1', 'from': 'q', 'to': 'qe'},
            {'id': 'b2', 'from': 'qe', 'to': 'near'},
            {'id': 'b3',
             'from': 'store',
             'to': 'near',
             'fs': 'b',
             'ts': 't',
             'ft': 0.3,
             'tt': 0.7,
             'my': 198},
            {'id': 'b4', 'from': 'near', 'to': 'ans'}],
  'steps': [{'n': ['files'], 'c': 'Index once: ovat index reads a folder of .md and .txt files.'},
            {'e': ['a1'],
             'n': ['chunks'],
             'c': 'Each file is cut into short passages, called chunks.'},
            {'e': ['a2'],
             'n': ['emb'],
             'c': 'A small model, bge-small, turns each chunk into 384 numbers that stand for its '
                  'meaning.'},
            {'e': ['a3'],
             'n': ['store'],
             'c': 'They are saved in one sqlite-vec file. If a file changes later, OVAT tells '
                  'you.'},
            {'n': ['q'], 'c': 'Then, for every question:', 'd': 2.2},
            {'e': ['b1'],
             'n': ['qe'],
             'c': 'the question is turned into 384 numbers the same way,'},
            {'e': ['b2', 'b3'],
             'n': ['near'],
             'c': 'the chunks whose numbers are closest are pulled from the index,'},
            {'e': ['b4'],
             'n': ['ans'],
             'c': 'and the model answers from them. OVAT prints the source files itself.',
             'd': 3.6}],
  'name': 'rag'},
 {'id': 'devices',
  'w': 880,
  'h': 400,
  'kicker': 'which device runs what',
  'title': 'Device routing',
  'desc': 'On an AI PC the agent model and vision go to the GPU, embeddings to the NPU, and speech '
          'to text to the CPU. With only a CPU, everything runs on the CPU.',
  'summary': 'AI PC: LLM and vision on GPU, embeddings on NPU, whisper on CPU. CPU only: '
             'everything on CPU.',
  'groups': [{'x': 24, 'y': 64, 'w': 520, 'h': 268, 'label': 'intel ai pc: cpu + gpu + npu'},
             {'x': 560, 'y': 64, 'w': 300, 'h': 268, 'label': 'cpu only: a mac, a dev laptop'}],
  'nodes': [{'id': 'llm',
             'x': 40,
             'y': 96,
             'w': 186,
             'h': 48,
             'title': 'Agent model',
             'sub': ['the LLM, int4']},
            {'id': 'vis',
             'x': 40,
             'y': 152,
             'w': 186,
             'h': 48,
             'title': 'Vision',
             'sub': ['describe_image']},
            {'id': 'emb',
             'x': 40,
             'y': 208,
             'w': 186,
             'h': 48,
             'title': 'Embeddings',
             'sub': ['bge-small, ~130 MB']},
            {'id': 'asr',
             'x': 40,
             'y': 264,
             'w': 186,
             'h': 48,
             'title': 'Speech to text',
             'sub': ['whisper, ~80 MB']},
            {'id': 'gpu',
             'x': 330,
             'y': 96,
             'w': 198,
             'h': 104,
             'title': 'GPU',
             'mono': True,
             'size': 15,
             'sub': ['dynamic shapes,', 'KV cache, works with', 'every export']},
            {'id': 'npu',
             'x': 330,
             'y': 208,
             'w': 198,
             'h': 48,
             'title': 'NPU',
             'mono': True,
             'size': 15,
             'sub': ['static shape, low power']},
            {'id': 'cpu',
             'x': 330,
             'y': 264,
             'w': 198,
             'h': 48,
             'title': 'CPU',
             'mono': True,
             'size': 15,
             'sub': ['small model, fast enough']},
            {'id': 'all',
             'x': 578,
             'y': 96,
             'w': 264,
             'h': 216,
             'title': 'CPU',
             'mono': True,
             'size': 15,
             'sub': ['everything runs here',
                     '',
                     'low-bit weights keep',
                     'the model in RAM',
                     '',
                     'a device set in',
                     'workflow.yml always wins']}],
  'edges': [{'id': 'd1', 'from': 'llm', 'to': 'gpu', 'tt': 0.23},
            {'id': 'd2', 'from': 'vis', 'to': 'gpu', 'tt': 0.77},
            {'id': 'd3', 'from': 'emb', 'to': 'npu'},
            {'id': 'd4', 'from': 'asr', 'to': 'cpu'}],
  'steps': [{'e': ['d1'],
             'n': ['llm', 'gpu'],
             'c': 'On an AI PC the agent model goes to the GPU, which handles dynamic shapes and '
                  'the KV cache.'},
            {'e': ['d2'],
             'n': ['vis', 'gpu'],
             'c': 'Vision follows the LLM. With a unified model it is the same weights.'},
            {'e': ['d3'],
             'n': ['emb', 'npu'],
             'c': 'Embeddings suit the NPU: a small model with a static shape, at low power.'},
            {'e': ['d4'],
             'n': ['asr', 'cpu'],
             'c': 'Whisper is small enough that the CPU is fast enough.'},
            {'n': ['all'],
             'c': 'With only a CPU, everything runs there. A device set in workflow.yml always '
                  'wins.',
             'd': 3.6}],
  'name': 'devices'},
 {'id': 'agent-tui',
  'w': 880,
  'h': 410,
  'kicker': 'the agent tui, three layers',
  'cmd': '$ ovat agent',
  'title': 'How the agent TUI is built',
  'desc': "You type in the terminal client. It sends a JSON-RPC request to OVAT's app server, "
          'which starts a turn on AgentSession. AgentSession runs the chosen engine on a worker '
          'thread; the engine asks the model on OVMS, and every tool call goes through one wrapper '
          'that reports events. The events travel back up the same layers and the client draws '
          'them.',
  'summary': 'Client sends JSON-RPC to the app server, which starts a turn on AgentSession. Tool '
             'events travel back up as cards.',
  'nodes': [{'id': 'you',
             'x': 24,
             'y': 90,
             'w': 120,
             'h': 36,
             'pill': True,
             'title': 'you type',
             'size': 13},
            {'id': 'client',
             'x': 190,
             'y': 76,
             'w': 300,
             'h': 64,
             'k': 'ovat/tui/vibe/',
             'title': 'Terminal client',
             'sub': ['keys, layout, draws the cards']},
            {'id': 'server',
             'x': 190,
             'y': 172,
             'w': 300,
             'h': 64,
             'k': 'ovat/tui/server/',
             'title': 'App server',
             'sub': ["answers from OVAT's own state"]},
            {'id': 'session',
             'x': 190,
             'y': 268,
             'w': 300,
             'h': 64,
             'k': 'ovat/agentic/',
             'title': 'AgentSession',
             'sub': ['runs the engine on a worker thread']},
            {'id': 'wrap',
             'x': 600,
             'y': 172,
             'w': 256,
             'h': 64,
             'k': 'events.observed',
             'title': 'Tool wrapper',
             'sub': ['every tool call, every engine']},
            {'id': 'ovms',
             'x': 600,
             'y': 268,
             'w': 256,
             'h': 64,
             'title': 'Engine + model',
             'sub': ['one of four engines, OVMS']}],
  'edges': [{'id': 't0', 'from': 'you', 'to': 'client', 'tt': 0.55},
            {'id': 't1',
             'from': 'client',
             'to': 'server',
             'fs': 'b',
             'ts': 't',
             'ft': 0.3,
             'tt': 0.3,
             'label': 'JSON-RPC'},
            {'id': 't2',
             'from': 'server',
             'to': 'session',
             'fs': 'b',
             'ts': 't',
             'ft': 0.3,
             'tt': 0.3,
             'label': 'start a turn'},
            {'id': 't3', 'from': 'session', 'to': 'ovms'},
            {'id': 't4', 'from': 'ovms', 'to': 'wrap', 'fs': 't', 'ts': 'b', 'label': 'tool call'},
            {'id': 't5', 'from': 'wrap', 'to': 'server', 'fs': 'l', 'ts': 'r', 'label': 'events'},
            {'id': 't6',
             'from': 'server',
             'to': 'client',
             'fs': 't',
             'ts': 'b',
             'ft': 0.72,
             'tt': 0.72,
             'label': 'cards'}],
  'steps': [{'n': ['you', 'client'],
             'c': "You type in the terminal client. It knows nothing about OVAT's engines."},
            {'e': ['t1'],
             'n': ['server'],
             'c': "It sends a JSON-RPC request to OVAT's own app server."},
            {'e': ['t2'],
             'n': ['session'],
             'c': 'The app server starts a turn on AgentSession, which has no UI code at all.'},
            {'e': ['t3'],
             'n': ['ovms'],
             'c': 'AgentSession runs the chosen engine on a worker thread. The engine asks the '
                  'model.'},
            {'e': ['t4'],
             'n': ['wrap'],
             'c': 'Every tool call passes through one wrapper, whichever engine is running.'},
            {'e': ['t5'],
             'n': ['server'],
             'c': 'The wrapper reports tool_started and tool_ended as events.'},
            {'e': ['t6'],
             'n': ['client'],
             'c': 'The app server turns events into cards, and the client draws them.',
             'd': 3.6}],
  'name': 'agent-tui'},
 {'id': 'subagents',
  'w': 880,
  'h': 400,
  'kicker': 'subagents',
  'title': 'Subagents: one tool, three kinds',
  'desc': 'The main agent gets one tool, delegate(agent, task). A subagent is another OVAT '
          'workflow in this process, a model behind an OpenAI-compatible API, or an agent reached '
          'over A2A. Answers come back framed as information. Progress is reported as events and '
          'shown in the /agents table.',
  'summary': 'delegate(agent, task) reaches a workflow, an API model or an A2A agent. Progress '
             'shows in /agents.',
  'nodes': [{'id': 'main',
             'x': 24,
             'y': 140,
             'w': 210,
             'h': 90,
             'k': 'main agent',
             'title': 'delegate(agent, task)',
             'mono': True,
             'size': 12.5,
             'sub': ['its one tool for subagents']},
            {'id': 'wf',
             'x': 320,
             'y': 72,
             'w': 262,
             'h': 66,
             'title': 'Another workflow',
             'sub': ['in this process, e.g. researcher']},
            {'id': 'api',
             'x': 320,
             'y': 152,
             'w': 262,
             'h': 66,
             'title': 'A model behind an API',
             'sub': ['any OpenAI-compatible URL']},
            {'id': 'a2a',
             'x': 320,
             'y': 232,
             'w': 262,
             'h': 66,
             'title': 'An agent over A2A',
             'sub': ['another machine, e.g. team_docs']},
            {'id': 'tab',
             'x': 660,
             'y': 120,
             'w': 200,
             'h': 130,
             'k': '/agents',
             'title': 'Live table',
             'sub': ['kind, status, time,', 'tokens', 'x cancels one child']}],
  'edges': [{'id': 's1', 'from': 'main', 'to': 'wf', 'mx': 280},
            {'id': 's2', 'from': 'main', 'to': 'api', 'mx': 280},
            {'id': 's3', 'from': 'main', 'to': 'a2a', 'mx': 280},
            {'id': 'p1', 'from': 'wf', 'to': 'tab', 'tt': 0.25, 'mx': 622},
            {'id': 'p2', 'from': 'api', 'to': 'tab', 'tt': 0.5, 'mx': 622},
            {'id': 'p3', 'from': 'a2a', 'to': 'tab', 'tt': 0.75, 'mx': 622},
            {'id': 'ret',
             'pts': [[451, 298], [451, 316], [129, 316], [129, 230]],
             'label': 'answer, framed as information',
             'lx': 300,
             'ly': 316}],
  'steps': [{'n': ['main'],
             'c': 'The main agent sees one tool for all its helpers: delegate(agent, task).'},
            {'e': ['s1'],
             'n': ['wf'],
             'c': 'A subagent can be another OVAT workflow, run in this same process.'},
            {'e': ['s2'],
             'n': ['api'],
             'c': 'Or a model behind any OpenAI-compatible API. The approval shows where the task '
                  'goes.'},
            {'e': ['s3'], 'n': ['a2a'], 'c': 'Or an agent on another machine, reached over A2A.'},
            {'e': ['p1', 'p2', 'p3'],
             'n': ['tab'],
             'c': 'Progress arrives as events in the /agents table. Press x to cancel one child.'},
            {'e': ['ret'],
             'n': ['main'],
             'c': 'The answer comes back framed as information, never as instructions.',
             'd': 3.6}],
  'name': 'subagents'},
 {'id': 'a2a-flow',
  'w': 880,
  'h': 420,
  'kicker': 'agent2agent, one request',
  'cmd': '$ ovat a2a serve workflow.yml',
  'title': 'One A2A request, from the caller to the answer',
  'desc': 'A caller sends a message. The server checks the host, the body size and the bearer '
          'token, then admission limits, then gives each caller and conversation its own agent. A '
          'run takes the conversation lock and then the one-model gate, runs only the exposed '
          'tools, and the answer comes back as a task update.',
  'summary': 'Host and size, token, admission, own agent; then lock, model gate, exposed tools '
             'only, answer as a task update.',
  'nodes': [{'id': 'caller',
             'x': 24,
             'y': 88,
             'w': 120,
             'h': 40,
             'pill': True,
             'title': 'caller',
             'size': 13},
            {'id': 'd1',
             'x': 180,
             'y': 72,
             'w': 150,
             'h': 72,
             'k': 'door 1',
             'title': 'Host + size',
             'sub': ['our Host header,', 'a small body']},
            {'id': 'd2',
             'x': 350,
             'y': 72,
             'w': 150,
             'h': 72,
             'k': 'door 2',
             'title': 'Bearer token',
             'sub': ['always, even on', '127.0.0.1']},
            {'id': 'd3',
             'x': 520,
             'y': 72,
             'w': 150,
             'h': 72,
             'k': 'door 3',
             'title': 'Admission',
             'sub': ['queue, rate,', 'turns, hops']},
            {'id': 'd4',
             'x': 690,
             'y': 72,
             'w': 166,
             'h': 72,
             'k': 'door 4',
             'title': 'Its own agent',
             'sub': ['per caller and', 'conversation']},
            {'id': 'lock',
             'x': 690,
             'y': 214,
             'w': 166,
             'h': 72,
             'title': 'Conversation lock',
             'sub': ['taken first']},
            {'id': 'gate',
             'x': 520,
             'y': 214,
             'w': 150,
             'h': 72,
             'title': 'One-model gate',
             'sub': ['tasks take turns', 'on one OVMS']},
            {'id': 'tools',
             'x': 350,
             'y': 214,
             'w': 150,
             'h': 72,
             'title': 'Exposed tools',
             'sub': ['the rest are', 'never built']},
            {'id': 'ans',
             'x': 180,
             'y': 214,
             'w': 150,
             'h': 72,
             'kind': 'ok',
             'title': 'Task update',
             'sub': ['the answer', {'c': 'audit.jsonl'}]}],
  'texts': [{'x': 180,
             'y': 172,
             'mono': True,
             's': 'any door can refuse. nothing past it runs.',
             'color': 'muted',
             'size': 11}],
  'edges': [{'id': 'c1', 'from': 'caller', 'to': 'd1', 'tt': 0.5},
            {'id': 'c2', 'from': 'd1', 'to': 'd2'},
            {'id': 'c3', 'from': 'd2', 'to': 'd3'},
            {'id': 'c4', 'from': 'd3', 'to': 'd4'},
            {'id': 'c5', 'from': 'd4', 'to': 'lock', 'fs': 'b', 'ts': 't'},
            {'id': 'c6', 'from': 'lock', 'to': 'gate', 'fs': 'l', 'ts': 'r'},
            {'id': 'c7', 'from': 'gate', 'to': 'tools', 'fs': 'l', 'ts': 'r'},
            {'id': 'c8', 'from': 'tools', 'to': 'ans', 'fs': 'l', 'ts': 'r'},
            {'id': 'c9', 'from': 'ans', 'to': 'caller', 'fs': 'l', 'ts': 'b'}],
  'steps': [{'n': ['caller'],
             'c': 'Another agent sends a message to the served workflow.',
             'd': 2.4},
            {'e': ['c1'],
             'n': ['d1'],
             'c': "Door 1: the Host header must be the server's own, and the body must be small."},
            {'e': ['c2'],
             'n': ['d2'],
             'c': 'Door 2: a bearer token, always. A web page can reach a loopback port too.'},
            {'e': ['c3'],
             'n': ['d3'],
             'c': 'Door 3: limits on queue, rate, turns and hops, before any model work starts.'},
            {'e': ['c4'],
             'n': ['d4'],
             'c': 'Each caller and conversation gets its own agent, so callers never share '
                  'memory.'},
            {'e': ['c5'], 'n': ['lock'], 'c': "The run takes its conversation's lock first,"},
            {'e': ['c6'],
             'n': ['gate'],
             'c': 'then the one-model gate. A waiting caller is told how many are ahead.'},
            {'e': ['c7'],
             'n': ['tools'],
             'c': 'Only exposed tools run. An unexposed tool is not even built into the agent.'},
            {'e': ['c8', 'c9'],
             'n': ['ans', 'caller'],
             'c': 'The answer goes back as a task update, and is written to the audit log.',
             'd': 3.6}],
  'name': 'a2a-flow'},
 {'id': 'approval-flow',
  'w': 880,
  'h': 400,
  'kicker': 'approvals',
  'title': 'How a tool call is approved',
  'desc': 'The model asks for a tool. One wrapper, events.observed, checks the permission of that '
          'tool or subagent. Allow runs it. Ask opens an approval panel with once, session, always '
          'and deny when someone is there to answer, and counts as no when nobody is. Deny sends '
          'the model a plain sentence.',
  'summary': 'One wrapper checks every tool: allow runs it, ask opens a panel (or means no), deny '
             'tells the model why.',
  'nodes': [{'id': 'model',
             'x': 24,
             'y': 150,
             'w': 150,
             'h': 48,
             'pill': True,
             'title': 'model asks',
             'size': 13},
            {'id': 'wrap',
             'x': 214,
             'y': 136,
             'w': 210,
             'h': 76,
             'k': 'events.observed',
             'title': 'Permission check',
             'sub': ['one wrapper, all engines']},
            {'id': 'allow',
             'x': 476,
             'y': 70,
             'w': 150,
             'h': 52,
             'title': 'allow',
             'mono': True,
             'sub': ['read, search']},
            {'id': 'ask',
             'x': 476,
             'y': 148,
             'w': 150,
             'h': 52,
             'kind': 'q',
             'title': 'ask',
             'mono': True,
             'sub': ['write, shell, web']},
            {'id': 'deny',
             'x': 476,
             'y': 262,
             'w': 150,
             'h': 52,
             'title': 'deny',
             'mono': True,
             'sub': ['set by you']},
            {'id': 'run',
             'x': 690,
             'y': 70,
             'w': 166,
             'h': 52,
             'kind': 'ok',
             'title': 'Tool runs',
             'sub': ['result to the model']},
            {'id': 'panel',
             'x': 690,
             'y': 142,
             'w': 166,
             'h': 64,
             'title': 'Approval panel',
             'sub': ['once, session,', 'always or deny']},
            {'id': 'no',
             'x': 690,
             'y': 262,
             'w': 166,
             'h': 52,
             'kind': 'fail',
             'title': 'Plain sentence',
             'sub': ['the model reads why']}],
  'edges': [{'id': 'a0', 'from': 'model', 'to': 'wrap'},
            {'id': 'a1', 'from': 'wrap', 'to': 'allow', 'mx': 450},
            {'id': 'a2', 'from': 'wrap', 'to': 'ask', 'mx': 450},
            {'id': 'a3', 'from': 'wrap', 'to': 'deny', 'mx': 450},
            {'id': 'a4', 'from': 'allow', 'to': 'run'},
            {'id': 'a5',
             'from': 'ask',
             'to': 'panel',
             'label': 'someone there',
             'lx': 658,
             'ly': 174},
            {'id': 'a6',
             'from': 'panel',
             'to': 'run',
             'fs': 't',
             'ts': 'b',
             'label': 'allowed',
             'lx': 773,
             'ly': 132},
            {'id': 'a7',
             'from': 'ask',
             'to': 'no',
             'fs': 'b',
             'ts': 'l',
             'ft': 0.5,
             'tt': 0.3,
             'label': 'nobody there',
             'lx': 551,
             'ly': 236},
            {'id': 'a8', 'from': 'deny', 'to': 'no'}],
  'steps': [{'n': ['model'], 'c': 'The model asks for a tool, or for a subagent.', 'd': 2.4},
            {'e': ['a0'],
             'n': ['wrap'],
             'c': "One wrapper checks that tool's permission. It sits under all four engines."},
            {'e': ['a1', 'a4'],
             'n': ['allow', 'run'],
             'c': 'allow: it runs. Reading files and searching your documents are allow by '
                  'default.'},
            {'e': ['a2', 'a5'],
             'n': ['ask', 'panel'],
             'c': 'ask: a panel opens. Allow once, for the session, always, or deny.'},
            {'e': ['a6'],
             'n': ['run'],
             'c': 'Allowed, it runs. "Always" is kept in ~/.ovat, never inside the project.'},
            {'e': ['a7'],
             'n': ['no'],
             'c': 'With nobody to ask (ovat run, bench, an A2A server), ask counts as no.'},
            {'e': ['a3', 'a8'],
             'n': ['deny', 'no'],
             'c': 'deny: the tool does not run, and the model is told so in one plain sentence.',
             'd': 3.6}],
  'name': 'approval-flow'}]

SPECS_ARCH = [{'id': 'stack',
  'w': 880,
  'h': 580,
  'group': 'arch',
  'kicker': 'the stack, top to bottom',
  'title': 'The OVAT stack',
  'desc': 'Six layers: interfaces, configuration, agent engines, providers and tools, serving and '
          'runtime, Intel hardware. Telemetry reads the engines, serving and hardware alongside '
          'the stack. A plano gateway is optional and outside OVAT.',
  'summary': 'Interfaces, config, engines, providers and tools, serving, hardware. Telemetry runs '
             'alongside.',
  'groups': [{'x': 24, 'y': 64, 'w': 636, 'h': 60, 'label': '1  interfaces'},
             {'x': 24, 'y': 140, 'w': 636, 'h': 60, 'label': '2  configuration'},
             {'x': 24, 'y': 216, 'w': 636, 'h': 60, 'label': '3  agent engines'},
             {'x': 24, 'y': 292, 'w': 636, 'h': 60, 'label': '4  providers and tools'},
             {'x': 24, 'y': 368, 'w': 636, 'h': 60, 'label': '5  serving and runtime'},
             {'x': 24, 'y': 444, 'w': 636, 'h': 60, 'label': '6  intel hardware'},
             {'x': 676, 'y': 64, 'w': 184, 'h': 364, 'label': 'telemetry, alongside'},
             {'x': 676, 'y': 444, 'w': 184, 'h': 60, 'label': 'optional, outside ovat'}],
  'nodes': [{'id': 'cli',
             'x': 40,
             'y': 88,
             'w': 130,
             'h': 30,
             'center': True,
             'title': 'ovat CLI',
             'mono': True,
             'optional': False,
             'size': 13},
            {'id': 'tui',
             'x': 182,
             'y': 88,
             'w': 130,
             'h': 30,
             'center': True,
             'title': 'Terminal UI',
             'mono': False,
             'optional': True,
             'size': 13},
            {'id': 'yaml',
             'x': 40,
             'y': 164,
             'w': 130,
             'h': 30,
             'center': True,
             'title': 'workflow.yml',
             'mono': True,
             'optional': False,
             'size': 13},
            {'id': 'schema',
             'x': 200,
             'y': 164,
             'w': 130,
             'h': 30,
             'center': True,
             'title': 'Strict schema',
             'mono': False,
             'optional': False,
             'size': 13},
            {'id': 'factory',
             'x': 360,
             'y': 164,
             'w': 130,
             'h': 30,
             'center': True,
             'title': 'Agent factory',
             'mono': False,
             'optional': False,
             'size': 13},
            {'id': 'native',
             'x': 40,
             'y': 240,
             'w': 130,
             'h': 30,
             'center': True,
             'title': 'native',
             'mono': True,
             'optional': False,
             'size': 13},
            {'id': 'react',
             'x': 182,
             'y': 240,
             'w': 130,
             'h': 30,
             'center': True,
             'title': 'react',
             'mono': True,
             'optional': False,
             'size': 13},
            {'id': 'llama',
             'x': 324,
             'y': 240,
             'w': 130,
             'h': 30,
             'center': True,
             'title': 'llamaindex',
             'mono': True,
             'optional': False,
             'size': 13},
            {'id': 'oai',
             'x': 466,
             'y': 240,
             'w': 130,
             'h': 30,
             'center': True,
             'title': 'openai-agents',
             'mono': True,
             'optional': False,
             'size': 13},
            {'id': 'llmp',
             'x': 40,
             'y': 316,
             'w': 130,
             'h': 30,
             'center': True,
             'title': 'LLM provider',
             'mono': False,
             'optional': False,
             'size': 13},
            {'id': 'emb',
             'x': 182,
             'y': 316,
             'w': 130,
             'h': 30,
             'center': True,
             'title': 'Embeddings, RAG',
             'mono': False,
             'optional': False,
             'size': 13},
            {'id': 'builtin',
             'x': 324,
             'y': 316,
             'w': 130,
             'h': 30,
             'center': True,
             'title': 'Built-in tools',
             'mono': False,
             'optional': False,
             'size': 13},
            {'id': 'mcp',
             'x': 466,
             'y': 316,
             'w': 130,
             'h': 30,
             'center': True,
             'title': 'MCP client',
             'mono': False,
             'optional': True,
             'size': 13},
            {'id': 'ovms',
             'x': 40,
             'y': 392,
             'w': 130,
             'h': 30,
             'center': True,
             'title': 'OVMS server',
             'mono': False,
             'optional': False,
             'size': 13},
            {'id': 'genai',
             'x': 182,
             'y': 392,
             'w': 150,
             'h': 30,
             'center': True,
             'title': 'openvino_genai',
             'mono': True,
             'optional': False,
             'size': 13},
            {'id': 'cpu',
             'x': 40,
             'y': 468,
             'w': 90,
             'h': 30,
             'center': True,
             'title': 'CPU',
             'mono': True,
             'optional': False,
             'size': 13},
            {'id': 'gpu',
             'x': 142,
             'y': 468,
             'w': 90,
             'h': 30,
             'center': True,
             'title': 'GPU',
             'mono': True,
             'optional': False,
             'size': 13},
            {'id': 'npu',
             'x': 244,
             'y': 468,
             'w': 90,
             'h': 30,
             'center': True,
             'title': 'NPU',
             'mono': True,
             'optional': False,
             'size': 13},
            {'id': 'src',
             'x': 688,
             'y': 88,
             'w': 160,
             'h': 140,
             'title': 'Sources',
             'sub': ['run trace', 'system, process', 'NPU', 'OVMS log', 'Intel UT']},
            {'id': 'snk',
             'x': 688,
             'y': 300,
             'w': 160,
             'h': 100,
             'title': 'Sinks',
             'sub': ['JSON Lines', 'TUI page']},
            {'id': 'plano',
             'x': 688,
             'y': 468,
             'w': 160,
             'h': 30,
             'center': True,
             'size': 13,
             'optional': True,
             'title': 'plano gateway'}],
  'edges': [{'id': 'c1', 'from': 'yaml', 'to': 'schema'},
            {'id': 'c2', 'from': 'schema', 'to': 'factory'},
            {'id': 's1', 'pts': [[628, 124], [628, 140]]},
            {'id': 's2', 'pts': [[628, 200], [628, 216]]},
            {'id': 's3', 'pts': [[628, 276], [628, 292]]},
            {'id': 's4', 'pts': [[628, 352], [628, 368]]},
            {'id': 's5', 'pts': [[628, 428], [628, 444]]},
            {'id': 't1', 'dashed': True, 'pts': [[660, 255], [668, 255], [668, 180], [688, 180]]},
            {'id': 't2', 'dashed': True, 'pts': [[660, 407], [668, 407], [668, 180], [688, 180]]},
            {'id': 't3', 'dashed': True, 'pts': [[660, 483], [668, 483], [668, 180], [688, 180]]},
            {'id': 'ts', 'from': 'src', 'to': 'snk', 'fs': 'b', 'ts': 't'}],
  'steps': [{'n': ['cli', 'tui'],
             'c': 'You start in the ovat CLI, or in the optional terminal UI.'},
            {'e': ['s1', 'c1', 'c2'],
             'n': ['yaml', 'schema', 'factory'],
             'c': 'workflow.yml passes a strict schema, then the agent factory builds the agent.'},
            {'e': ['s2'],
             'n': ['native', 'react', 'llama', 'oai'],
             'c': 'agent.type picks one of four engines to run the loop.'},
            {'e': ['s3'],
             'n': ['llmp', 'emb', 'builtin', 'mcp'],
             'c': 'Engines call the LLM provider, document search, built-in tools and MCP '
                  'servers.'},
            {'e': ['s4'],
             'n': ['ovms', 'genai'],
             'c': 'The model is served by OVMS, or in-process by openvino_genai.'},
            {'e': ['s5'],
             'n': ['cpu', 'gpu', 'npu'],
             'c': 'Both run on Intel hardware: the CPU, GPU or NPU.'},
            {'e': ['t1', 't2', 't3', 'ts'],
             'n': ['src', 'snk'],
             'c': 'Telemetry reads the engines, serving and hardware, and writes JSON Lines or a '
                  'TUI page.'},
            {'n': ['plano'],
             'c': 'Optionally, a plano gateway in front of OVMS records OpenTelemetry spans.',
             'd': 3.6}],
  'name': 'stack'},
 {'id': 'run-sequence',
  'w': 880,
  'h': 590,
  'group': 'arch',
  'kicker': 'one ovat run, step by step',
  'cmd': '$ ovat run workflow.yml -i "question"',
  'title': 'One ovat run, step by step',
  'desc': 'The CLI loads and validates workflow.yml, the agent factory builds the agent, and the '
          'engine loops: it sends the history and tool schemas to OVMS, runs any tool the reply '
          'asks for and adds the result, until a reply has no tool calls. The answer, a sources '
          'line and an exit code come back.',
  'summary': 'Load, build, then loop with OVMS and the tools until a reply has no tool calls. '
             'Answer, sources, exit code.',
  'groups': [{'x': 476, 'y': 308, 'w': 384, 'h': 132, 'label': 'loop, max_iterations'}],
  'lines': [{'x1': 70, 'y1': 104, 'x2': 70, 'y2': 520},
            {'x1': 190, 'y1': 104, 'x2': 190, 'y2': 520},
            {'x1': 310, 'y1': 104, 'x2': 310, 'y2': 520},
            {'x1': 430, 'y1': 104, 'x2': 430, 'y2': 520},
            {'x1': 560, 'y1': 104, 'x2': 560, 'y2': 520},
            {'x1': 690, 'y1': 104, 'x2': 690, 'y2': 520},
            {'x1': 810, 'y1': 104, 'x2': 810, 'y2': 520}],
  'nodes': [{'id': 'user',
             'x': 24,
             'y': 62,
             'w': 92,
             'h': 40,
             'pill': True,
             'title': 'you',
             'size': 13},
            {'id': 'cli',
             'x': 136,
             'y': 60,
             'w': 108,
             'h': 44,
             'center': True,
             'title': 'ovat CLI',
             'mono': True,
             'size': 12},
            {'id': 'config',
             'x': 256,
             'y': 60,
             'w': 108,
             'h': 44,
             'center': True,
             'title': 'Config',
             'mono': True,
             'size': 12},
            {'id': 'factory',
             'x': 376,
             'y': 60,
             'w': 108,
             'h': 44,
             'center': True,
             'title': 'factory',
             'mono': True,
             'size': 12},
            {'id': 'engine',
             'x': 506,
             'y': 60,
             'w': 108,
             'h': 44,
             'center': True,
             'title': 'Engine',
             'mono': True,
             'size': 12},
            {'id': 'tool',
             'x': 636,
             'y': 60,
             'w': 108,
             'h': 44,
             'center': True,
             'title': 'Tool',
             'mono': True,
             'size': 12},
            {'id': 'ovms',
             'x': 756,
             'y': 60,
             'w': 108,
             'h': 44,
             'center': True,
             'title': 'OVMS /v3',
             'mono': True,
             'size': 12}],
  'edges': [{'id': 'm1', 'pts': [[74, 134], [184, 134]], 'label': 'ovat run', 'lx': 130, 'ly': 121},
            {'id': 'm2', 'pts': [[194, 166], [304, 166]], 'label': 'load', 'lx': 250, 'ly': 153},
            {'id': 'm3',
             'pts': [[306, 198], [196, 198]],
             'label': 'WorkflowConfig',
             'lx': 250,
             'ly': 185,
             'dashed': True},
            {'id': 'm4',
             'pts': [[194, 230], [424, 230]],
             'label': 'build_agent',
             'lx': 310,
             'ly': 217},
            {'id': 'm5',
             'pts': [[426, 262], [196, 262]],
             'label': 'agent',
             'lx': 310,
             'ly': 249,
             'dashed': True},
            {'id': 'm6',
             'pts': [[194, 294], [554, 294]],
             'label': 'agent.run',
             'lx': 375,
             'ly': 281},
            {'id': 'm7',
             'pts': [[564, 352], [804, 352]],
             'label': 'history + tools',
             'lx': 685,
             'ly': 339},
            {'id': 'm8',
             'pts': [[806, 376], [566, 376]],
             'label': 'reply',
             'lx': 685,
             'ly': 363,
             'dashed': True},
            {'id': 'm9',
             'pts': [[564, 406], [684, 406]],
             'label': 'tool call',
             'lx': 625,
             'ly': 393},
            {'id': 'm10',
             'pts': [[686, 434], [566, 434]],
             'label': 'result',
             'lx': 625,
             'ly': 421,
             'dashed': True},
            {'id': 'm11',
             'pts': [[556, 470], [196, 470]],
             'label': 'answer + trace',
             'lx': 375,
             'ly': 457,
             'dashed': True},
            {'id': 'm12',
             'pts': [[186, 502], [76, 502]],
             'label': 'answer',
             'lx': 130,
             'ly': 489,
             'dashed': True}],
  'steps': [{'e': ['m1'],
             'n': ['user', 'cli'],
             'c': 'You run ovat run with a workflow file and a question.',
             'd': 2.6},
            {'e': ['m2'],
             'n': ['config'],
             'c': 'The CLI loads the file through load_workflow().',
             'd': 2.6},
            {'e': ['m3'],
             'n': ['cli'],
             'c': 'It gets back a WorkflowConfig, or one sentence naming the error.',
             'd': 2.6},
            {'e': ['m4'],
             'n': ['factory'],
             'c': 'build_agent() wires the engine, the LLM provider and the tools.',
             'd': 2.6},
            {'e': ['m5'], 'n': ['cli'], 'c': 'The agent comes back to the CLI.', 'd': 2.2},
            {'e': ['m6'], 'n': ['engine'], 'c': 'agent.run(question) starts the loop.', 'd': 2.2},
            {'e': ['m7'],
             'n': ['ovms'],
             'c': 'Each round, the engine sends the history and the tool schemas to OVMS.',
             'd': 2.6},
            {'e': ['m8'],
             'n': ['engine'],
             'c': 'OVMS replies. If the reply carries tool calls, the loop goes on.',
             'd': 2.6},
            {'e': ['m9'],
             'n': ['tool'],
             'c': 'The engine runs the tool with the parsed arguments.',
             'd': 2.6},
            {'e': ['m10'],
             'n': ['engine'],
             'c': 'The result, or an "Error: ..." string, is added to the history.',
             'd': 2.6},
            {'e': ['m11'],
             'n': ['cli'],
             'c': 'A reply with no tool calls is the answer. It goes back to the CLI.',
             'd': 2.6},
            {'e': ['m12'],
             'n': ['user'],
             'c': 'You get the answer, a sources line and exit code 0 or 1.',
             'd': 3.4}],
  'name': 'run-sequence'},
 {'id': 'load-config',
  'w': 880,
  'h': 360,
  'group': 'arch',
  'kicker': 'loading a workflow',
  'title': 'Loading a workflow',
  'desc': 'Every command loads workflow.yml through _load_config(), the only loader: '
          'yaml.safe_load parses it, WorkflowConfig validates it with extra=forbid, a model '
          'validator fills tool_parser from the model name, and build_agent() receives it. A '
          'missing file, bad YAML or a schema error each become one sentence.',
  'summary': 'One loader: parse, validate with extra=forbid, fill tool_parser, build. Any failure '
             'is one sentence.',
  'nodes': [{'id': 'yaml',
             'x': 24,
             'y': 88,
             'w': 130,
             'h': 76,
             'title': 'workflow.yml',
             'mono': True,
             'size': 12,
             'sub': ['the file']},
            {'id': 'load',
             'x': 165,
             'y': 88,
             'w': 130,
             'h': 76,
             'title': '_load_config()',
             'mono': True,
             'size': 11.5,
             'sub': ['the only loader']},
            {'id': 'parse',
             'x': 306,
             'y': 88,
             'w': 130,
             'h': 76,
             'title': 'safe_load',
             'mono': True,
             'size': 12,
             'sub': ['or {} if empty']},
            {'id': 'val',
             'x': 447,
             'y': 88,
             'w': 130,
             'h': 76,
             'title': 'WorkflowConfig',
             'mono': True,
             'size': 11.5,
             'sub': ['extra=forbid']},
            {'id': 'der',
             'x': 588,
             'y': 88,
             'w': 130,
             'h': 76,
             'title': 'Derive',
             'sub': ['tool_parser from', 'the model name']},
            {'id': 'fac',
             'x': 729,
             'y': 88,
             'w': 130,
             'h': 76,
             'kind': 'ok',
             'title': 'build_agent()',
             'mono': True,
             'size': 12,
             'sub': ['a config it', 'can trust']},
            {'id': 'f1',
             'x': 100,
             'y': 232,
             'w': 136,
             'h': 56,
             'kind': 'fail',
             'title': 'Missing file',
             'sub': ['one sentence']},
            {'id': 'f2',
             'x': 250,
             'y': 232,
             'w': 136,
             'h': 56,
             'kind': 'fail',
             'title': 'Bad YAML',
             'sub': ['one sentence']},
            {'id': 'f3',
             'x': 400,
             'y': 232,
             'w': 136,
             'h': 56,
             'kind': 'fail',
             'title': 'Schema error',
             'sub': ['one sentence']}],
  'edges': [{'id': 'e1', 'from': 'yaml', 'to': 'load'},
            {'id': 'e2', 'from': 'load', 'to': 'parse'},
            {'id': 'e3', 'from': 'parse', 'to': 'val'},
            {'id': 'e4', 'from': 'val', 'to': 'der'},
            {'id': 'e5', 'from': 'der', 'to': 'fac'},
            {'id': 'x1',
             'from': 'load',
             'to': 'f1',
             'fs': 'b',
             'ts': 't',
             'ft': 0.3,
             'my': 204,
             'dashed': True},
            {'id': 'x2',
             'from': 'load',
             'to': 'f2',
             'fs': 'b',
             'ts': 't',
             'ft': 0.5,
             'my': 204,
             'dashed': True},
            {'id': 'x3',
             'from': 'load',
             'to': 'f3',
             'fs': 'b',
             'ts': 't',
             'ft': 0.7,
             'my': 204,
             'dashed': True}],
  'steps': [{'n': ['yaml'], 'c': 'Every command reads workflow.yml the same way.'},
            {'e': ['e1'],
             'n': ['load'],
             'c': '_load_config() is the only loader. No command reads the YAML on its own.'},
            {'e': ['e2'],
             'n': ['parse'],
             'c': 'yaml.safe_load parses it. An empty file becomes {}.'},
            {'e': ['e3'],
             'n': ['val'],
             'c': 'WorkflowConfig validates it. With extra=forbid, an unknown key is an error.'},
            {'e': ['e4'],
             'n': ['der'],
             'c': 'A model validator fills tool_parser from the model name.'},
            {'e': ['e5'], 'n': ['fac'], 'c': 'build_agent() gets a config it can trust.'},
            {'e': ['x1', 'x2', 'x3'],
             'n': ['f1', 'f2', 'f3'],
             'c': 'A missing file, bad YAML or a schema error each end in one sentence.',
             'd': 3.6}],
  'name': 'load-config'},
 {'id': 'loop-exits',
  'w': 960,
  'h': 450,
  'group': 'arch',
  'kicker': 'every exit from the native loop',
  'title': 'Every exit from the native loop',
  'desc': 'Ask, read, act, report, until no tool calls come back or the round cap is hit. A reply '
          'labelled tool_calls with none sent is reported, not re-asked. Leftover markup is '
          'undecoded_tool_call, reasoning only is empty_answer, and a real answer comes back with '
          'the trace.',
  'summary': 'Ask, read, act, report. Exits: the cap, a label with no calls, undecoded markup, '
             'empty answer, or the answer.',
  'nodes': [{'id': 'start',
             'x': 24,
             'y': 96,
             'w': 140,
             'h': 40,
             'pill': True,
             'title': 'user message',
             'size': 13},
            {'id': 'ask',
             'x': 188,
             'y': 84,
             'w': 150,
             'h': 64,
             'k': '1 ask',
             'title': 'POST history',
             'sub': ['and the tool menu']},
            {'id': 'read',
             'x': 362,
             'y': 84,
             'w': 170,
             'h': 64,
             'kind': 'q',
             'k': '2 read',
             'title': 'Tool calls?'},
            {'id': 'label',
             'x': 556,
             'y': 84,
             'w': 180,
             'h': 64,
             'kind': 'q',
             'title': 'finish_reason',
             'sub': ['says tool_calls?']},
            {'id': 'errA',
             'x': 776,
             'y': 84,
             'w': 160,
             'h': 64,
             'kind': 'fail',
             'title': 'Report it',
             'sub': ['none sent, no re-ask']},
            {'id': 'cap',
             'x': 24,
             'y': 196,
             'w': 140,
             'h': 64,
             'kind': 'q',
             'k': 'cap',
             'title': 'Rounds left?'},
            {'id': 'report',
             'x': 188,
             'y': 196,
             'w': 150,
             'h': 64,
             'k': '4 report',
             'title': 'Add results',
             'sub': ['to the history']},
            {'id': 'act',
             'x': 362,
             'y': 196,
             'w': 170,
             'h': 64,
             'k': '3 act',
             'title': 'Run each tool'},
            {'id': 'check',
             'x': 556,
             'y': 196,
             'w': 180,
             'h': 64,
             'kind': 'q',
             'title': 'Content usable?'},
            {'id': 'errB',
             'x': 24,
             'y': 310,
             'w': 140,
             'h': 56,
             'kind': 'fail',
             'title': 'Report the cap'},
            {'id': 'errC',
             'x': 362,
             'y': 310,
             'w': 170,
             'h': 56,
             'kind': 'fail',
             'title': 'undecoded_tool_call',
             'mono': True,
             'size': 10.5,
             'sub': ['markup left in it']},
            {'id': 'errD',
             'x': 556,
             'y': 310,
             'w': 180,
             'h': 56,
             'kind': 'fail',
             'title': 'empty_answer',
             'mono': True,
             'size': 12,
             'sub': ['only reasoning']},
            {'id': 'done',
             'x': 776,
             'y': 310,
             'w': 160,
             'h': 56,
             'kind': 'ok',
             'title': 'answer + trace',
             'size': 13}],
  'edges': [{'id': 'sa', 'from': 'start', 'to': 'ask'},
            {'id': 'ar', 'from': 'ask', 'to': 'read'},
            {'id': 'ra', 'from': 'read', 'to': 'act', 'fs': 'b', 'ts': 't', 'label': 'yes'},
            {'id': 'ap', 'from': 'act', 'to': 'report', 'fs': 'l', 'ts': 'r'},
            {'id': 'pc', 'from': 'report', 'to': 'cap', 'fs': 'l', 'ts': 'r'},
            {'id': 'ca',
             'from': 'cap',
             'to': 'ask',
             'fs': 't',
             'ts': 'b',
             'tt': 0.2,
             'my': 172,
             'label': 'yes',
             'lx': 94,
             'ly': 172},
            {'id': 'cb', 'from': 'cap', 'to': 'errB', 'fs': 'b', 'ts': 't', 'label': 'no'},
            {'id': 'rl', 'from': 'read', 'to': 'label', 'label': 'no', 'lx': 544, 'ly': 100},
            {'id': 'la', 'from': 'label', 'to': 'errA', 'label': 'yes', 'lx': 756, 'ly': 100},
            {'id': 'lc', 'from': 'label', 'to': 'check', 'fs': 'b', 'ts': 't', 'label': 'no'},
            {'id': 'kc', 'from': 'check', 'to': 'errC', 'fs': 'b', 'ts': 't', 'ft': 0.2, 'my': 286},
            {'id': 'kd', 'from': 'check', 'to': 'errD', 'fs': 'b', 'ts': 't'},
            {'id': 'kn', 'from': 'check', 'to': 'done', 'fs': 'r', 'ts': 't'}],
  'steps': [{'n': ['start'], 'c': 'A user message starts the loop.', 'd': 2.2},
            {'e': ['sa'],
             'n': ['ask'],
             'c': '1. Ask: POST the history and the tool menu to the model.',
             'd': 2.6},
            {'e': ['ar'],
             'n': ['read'],
             'c': '2. Read: are there tool calls in the reply?',
             'd': 2.6},
            {'e': ['ra'], 'n': ['act'], 'c': 'Yes: 3. Act. Each tool runs.', 'd': 2.4},
            {'e': ['ap'],
             'n': ['report'],
             'c': '4. Report: the tool results are added to the history.',
             'd': 2.6},
            {'e': ['pc'], 'n': ['cap'], 'c': 'Are there rounds left?', 'd': 2.2},
            {'e': ['ca'],
             'n': ['ask'],
             'c': 'Yes: ask again. The loop runs up to max_iterations rounds.',
             'd': 2.6},
            {'e': ['cb'], 'n': ['errB'], 'c': 'No rounds left: OVAT reports the cap.', 'd': 2.6},
            {'e': ['rl'],
             'n': ['label'],
             'c': 'No tool calls. Does finish_reason still say tool_calls?',
             'd': 2.6},
            {'e': ['la'],
             'n': ['errA'],
             'c': 'Yes, but none were sent: report it, do not re-ask.',
             'd': 2.6},
            {'e': ['lc'], 'n': ['check'], 'c': 'No: is the content usable?', 'd': 2.2},
            {'e': ['kc'],
             'n': ['errC'],
             'c': 'Tool markup left in the text: undecoded_tool_call.',
             'd': 2.6},
            {'e': ['kd'], 'n': ['errD'], 'c': 'Only reasoning, no answer: empty_answer.', 'd': 2.6},
            {'e': ['kn'],
             'n': ['done'],
             'c': 'A real answer comes back with the trace.',
             'd': 3.4}],
  'name': 'loop-exits'},
 {'id': 'providers',
  'w': 880,
  'h': 400,
  'group': 'arch',
  'kicker': 'the provider interfaces',
  'title': 'The provider interfaces',
  'desc': 'Four interfaces with their backends. LLMProvider: OVMS over the OpenAI SDK, which does '
          'tool calling, or openvino_genai in-process, which does not. EmbeddingsProvider: a local '
          'TextEmbeddingPipeline or OVMS /v3/embeddings. RetrieverProvider: sqlite-vec on disk or '
          'numpy in memory. VLMProvider: openvino_genai VLMPipeline.',
  'summary': 'LLM, embeddings, retriever, vision: four interfaces, each with its own backends.',
  'groups': [{'x': 24, 'y': 64, 'w': 200, 'h': 270, 'label': 'llmprovider'},
             {'x': 236, 'y': 64, 'w': 200, 'h': 270, 'label': 'embeddingsprovider'},
             {'x': 448, 'y': 64, 'w': 200, 'h': 270, 'label': 'retrieverprovider'},
             {'x': 660, 'y': 64, 'w': 200, 'h': 270, 'label': 'vlmprovider'}],
  'nodes': [{'id': 'l1',
             'x': 36,
             'y': 92,
             'w': 176,
             'h': 104,
             'title': 'OVMS',
             'mono': False,
             'size': 14,
             'sub': ['OpenAI SDK to /v3', 'does tool calling']},
            {'id': 'l2',
             'x': 36,
             'y': 212,
             'w': 176,
             'h': 104,
             'title': 'openvino_genai',
             'mono': True,
             'size': 13,
             'sub': ['in-process', 'no tool calling']},
            {'id': 'm1',
             'x': 248,
             'y': 92,
             'w': 176,
             'h': 104,
             'title': 'Local',
             'mono': False,
             'size': 14,
             'sub': ['TextEmbeddingPipeline', 'on this machine']},
            {'id': 'm2',
             'x': 248,
             'y': 212,
             'w': 176,
             'h': 104,
             'title': 'OVMS',
             'mono': False,
             'size': 14,
             'sub': [{'c': '/v3/embeddings'}]},
            {'id': 'r1',
             'x': 460,
             'y': 92,
             'w': 176,
             'h': 104,
             'title': 'sqlite-vec',
             'mono': True,
             'size': 13,
             'sub': ['saved to disk']},
            {'id': 'r2',
             'x': 460,
             'y': 212,
             'w': 176,
             'h': 104,
             'title': 'In memory',
             'mono': False,
             'size': 14,
             'sub': ['numpy,', 'nothing on disk']},
            {'id': 'v1',
             'x': 672,
             'y': 92,
             'w': 176,
             'h': 104,
             'title': 'openvino_genai',
             'mono': True,
             'size': 13,
             'sub': ['VLMPipeline']}],
  'steps': [{'n': ['l1', 'l2'],
             'c': 'LLMProvider: OVMS over the OpenAI SDK, or openvino_genai in-process. Only OVMS '
                  'calls tools.'},
            {'n': ['m1', 'm2'],
             'c': 'EmbeddingsProvider: a local TextEmbeddingPipeline, or OVMS /v3/embeddings.'},
            {'n': ['r1', 'r2'],
             'c': 'RetrieverProvider: sqlite-vec saved to disk, or numpy with nothing on disk.'},
            {'n': ['v1'], 'c': 'VLMProvider: the openvino_genai VLMPipeline.', 'd': 3.4}],
  'name': 'providers'},
 {'id': 'tool-source',
  'w': 880,
  'h': 440,
  'group': 'arch',
  'kicker': 'where a tool comes from',
  'title': 'Where a tool comes from',
  'desc': 'A tool is either builtin, with its schema from SCHEMA and a function bound by the '
          'factory, or mcp_stdio, where MCPStdioServer launches the server, lists its tools and '
          'calls them. Both reach the agent loop in the same shape, so the loop cannot tell them '
          'apart.',
  'summary': 'builtin or mcp_stdio. Both arrive in the same shape, so the loop treats them the '
             'same.',
  'nodes': [{'id': 'need',
             'x': 24,
             'y': 150,
             'w': 170,
             'h': 40,
             'pill': True,
             'title': 'agent needs a tool',
             'size': 13},
            {'id': 'type', 'x': 232, 'y': 140, 'w': 120, 'h': 60, 'kind': 'q', 'title': 'type?'},
            {'id': 'b',
             'x': 420,
             'y': 64,
             'w': 210,
             'h': 92,
             'title': 'builtin',
             'mono': True,
             'sub': ['schema from SCHEMA,', 'function bound by', 'the factory']},
            {'id': 'm',
             'x': 420,
             'y': 184,
             'w': 210,
             'h': 92,
             'title': 'mcp_stdio',
             'mono': True,
             'sub': ['MCPStdioServer:', 'launch, list_tools,', 'call_tool']},
            {'id': 'loop', 'x': 700, 'y': 140, 'w': 160, 'h': 60, 'title': 'Agent loop'},
            {'id': 'note',
             'x': 420,
             'y': 300,
             'w': 210,
             'h': 52,
             'optional': True,
             'title': 'Same shape',
             'sub': ['loop cannot tell them apart']}],
  'edges': [{'id': 'nt', 'from': 'need', 'to': 'type'},
            {'id': 'tb', 'from': 'type', 'to': 'b', 'ft': 0.3, 'mx': 386},
            {'id': 'tm', 'from': 'type', 'to': 'm', 'ft': 0.7, 'mx': 386},
            {'id': 'bl', 'from': 'b', 'to': 'loop', 'tt': 0.33, 'mx': 665},
            {'id': 'ml', 'from': 'm', 'to': 'loop', 'tt': 0.67, 'mx': 665},
            {'id': 'mn', 'from': 'm', 'to': 'note', 'fs': 'b', 'ts': 't', 'dashed': True}],
  'steps': [{'n': ['need'], 'c': 'The agent needs a tool. Where does it come from?', 'd': 2.6},
            {'e': ['nt'], 'n': ['type'], 'c': 'Each tool has a type: builtin or mcp_stdio.'},
            {'e': ['tb'],
             'n': ['b'],
             'c': 'builtin: the schema comes from SCHEMA and the factory binds the function.'},
            {'e': ['bl'], 'n': ['loop'], 'c': 'It goes into the agent loop.', 'd': 2.4},
            {'e': ['tm'],
             'n': ['m'],
             'c': 'mcp_stdio: MCPStdioServer launches the server, lists its tools and calls them.'},
            {'e': ['ml'], 'n': ['loop'], 'c': 'It goes into the same loop.', 'd': 2.4},
            {'e': ['mn'],
             'n': ['note'],
             'c': 'Both have the same shape, so the loop cannot tell them apart.',
             'd': 3.6}],
  'name': 'tool-source'},
 {'id': 'setup',
  'w': 960,
  'h': 510,
  'group': 'arch',
  'kicker': 'what ovat setup does',
  'cmd': '$ ovat setup',
  'title': 'What ovat setup does',
  'desc': 'ovat setup checks the platform: macOS has no build and exits 0, Windows gets the '
          'Python-enabled zip, Linux reads /etc/os-release and falls back to ubuntu24 with a '
          'warning. It downloads the archive, checks the SHA-256 when one is published, refuses on '
          'a mismatch, then extracts into ~/.ovat/ovms.',
  'summary': 'Pick the build for the platform, download, verify the SHA-256 when published, '
             'extract to ~/.ovat/ovms.',
  'nodes': [{'id': 's',
             'x': 24,
             'y': 158,
             'w': 120,
             'h': 40,
             'pill': True,
             'title': 'ovat setup',
             'mono': True,
             'size': 12.5},
            {'id': 'p', 'x': 168, 'y': 150, 'w': 120, 'h': 56, 'kind': 'q', 'title': 'Platform?'},
            {'id': 'mac',
             'x': 320,
             'y': 64,
             'w': 200,
             'h': 56,
             'kind': 'warn',
             'title': 'darwin',
             'mono': True,
             'sub': ['no macOS build, exit 0']},
            {'id': 'win',
             'x': 320,
             'y': 150,
             'w': 200,
             'h': 56,
             'title': 'win32',
             'mono': True,
             'sub': [{'c': 'ovms_windows_*.zip'}]},
            {'id': 'lin',
             'x': 320,
             'y': 236,
             'w': 200,
             'h': 72,
             'title': 'linux',
             'mono': True,
             'sub': ['reads /etc/os-release:', 'ubuntu22, 24, redhat']},
            {'id': 'warn',
             'x': 320,
             'y': 330,
             'w': 200,
             'h': 56,
             'kind': 'warn',
             'title': 'Unknown distro',
             'sub': ['warn, try ubuntu24']},
            {'id': 'dl', 'x': 556, 'y': 150, 'w': 160, 'h': 56, 'title': 'Download'},
            {'id': 'sum',
             'x': 752,
             'y': 150,
             'w': 184,
             'h': 56,
             'kind': 'q',
             'title': '.sha256 published?',
             'size': 13},
            {'id': 'unv',
             'x': 752,
             'y': 64,
             'w': 184,
             'h': 56,
             'kind': 'warn',
             'title': 'Unverified',
             'sub': ['installed-unverified']},
            {'id': 'sha',
             'x': 752,
             'y': 236,
             'w': 184,
             'h': 56,
             'kind': 'q',
             'title': 'SHA-256 matches?',
             'size': 13},
            {'id': 'stop',
             'x': 752,
             'y': 322,
             'w': 184,
             'h': 56,
             'kind': 'fail',
             'title': 'Refuse',
             'sub': ['install nothing']},
            {'id': 'ex',
             'x': 556,
             'y': 236,
             'w': 160,
             'h': 100,
             'title': 'Extract',
             'sub': ['flatten ovms/,', 'make every member', 'owner-writable']},
            {'id': 'r',
             'x': 536,
             'y': 366,
             'w': 200,
             'h': 64,
             'kind': 'ok',
             'title': '~/.ovat/ovms',
             'mono': True,
             'sub': ['bin/ovms or ovms.exe']}],
  'edges': [{'id': 'sp', 'from': 's', 'to': 'p'},
            {'id': 'pm', 'from': 'p', 'to': 'mac', 'ft': 0.3, 'mx': 304},
            {'id': 'pw', 'from': 'p', 'to': 'win'},
            {'id': 'pl', 'from': 'p', 'to': 'lin', 'ft': 0.7, 'mx': 304},
            {'id': 'lw', 'from': 'lin', 'to': 'warn', 'fs': 'b', 'ts': 't'},
            {'id': 'wd', 'from': 'win', 'to': 'dl'},
            {'id': 'ld', 'from': 'lin', 'to': 'dl', 'tt': 0.75, 'mx': 538},
            {'id': 'wad', 'from': 'warn', 'to': 'dl', 'tt': 0.9, 'mx': 538},
            {'id': 'ds', 'from': 'dl', 'to': 'sum'},
            {'id': 'su', 'from': 'sum', 'to': 'unv', 'fs': 't', 'ts': 'b', 'label': 'no'},
            {'id': 'ss', 'from': 'sum', 'to': 'sha', 'fs': 'b', 'ts': 't', 'label': 'yes'},
            {'id': 'sst', 'from': 'sha', 'to': 'stop', 'fs': 'b', 'ts': 't', 'label': 'no'},
            {'id': 'se',
             'from': 'sha',
             'to': 'ex',
             'fs': 'l',
             'ts': 'r',
             'tt': 0.28,
             'label': 'yes',
             'lx': 726,
             'ly': 264},
            {'id': 'ue', 'pts': [[752, 92], [748, 92], [748, 222], [636, 222], [636, 236]]},
            {'id': 'er', 'from': 'ex', 'to': 'r', 'fs': 'b', 'ts': 't'}],
  'steps': [{'n': ['s'], 'c': 'ovat setup installs the OVMS server for this machine.', 'd': 2.6},
            {'e': ['sp'], 'n': ['p'], 'c': 'It checks the platform.', 'd': 2.2},
            {'e': ['pm'], 'n': ['mac'], 'c': 'macOS: there is no build. It says so and exits 0.'},
            {'e': ['pw'], 'n': ['win'], 'c': 'Windows: the OVMS Windows zip with Python enabled.'},
            {'e': ['pl'],
             'n': ['lin'],
             'c': 'Linux: it reads /etc/os-release for ubuntu22, ubuntu24 or redhat.'},
            {'e': ['lw'],
             'n': ['warn'],
             'c': 'An unknown distro gets a warning, then the ubuntu24 build.'},
            {'e': ['wd', 'ld', 'wad'],
             'n': ['dl'],
             'c': 'Then it downloads the archive.',
             'd': 2.4},
            {'e': ['ds'], 'n': ['sum'], 'c': 'Is a .sha256 checksum published for it?', 'd': 2.6},
            {'e': ['su'],
             'n': ['unv'],
             'c': 'No: there is nothing to check. It is reported installed-unverified.'},
            {'e': ['ss'], 'n': ['sha'], 'c': 'Yes: does the SHA-256 match?', 'd': 2.4},
            {'e': ['sst'], 'n': ['stop'], 'c': 'No: it refuses and installs nothing.'},
            {'e': ['se', 'ue'],
             'n': ['ex'],
             'c': 'Yes: it extracts, flattens the ovms/ wrapper and makes every member '
                  'owner-writable.'},
            {'e': ['er'],
             'n': ['r'],
             'c': 'OVMS lands in ~/.ovat/ovms: bin/ovms on Linux, ovms.exe on Windows.',
             'd': 3.6}],
  'name': 'setup'},
 {'id': 'serve',
  'w': 960,
  'h': 484,
  'group': 'arch',
  'kicker': 'what ovat serve does',
  'cmd': '$ ovat serve',
  'title': 'What ovat serve does',
  'desc': 'find_ovms() looks in the config, OVAT_OVMS, PATH and known folders. If OVMS is missing '
          'it offers ovat setup at a terminal and refuses otherwise. If found, ovms_env() sets the '
          'environment, Popen starts OVMS with logs to a file, and wait_until_ready() watches for '
          'progress: ready on health 200, a report if the process exits, a still-running note '
          'after 300 s with no progress.',
  'summary': 'Find OVMS, set its environment, start it, then wait on progress: ready, exited, or '
             'still running.',
  'nodes': [{'id': 'sv',
             'x': 24,
             'y': 92,
             'w': 120,
             'h': 40,
             'pill': True,
             'title': 'ovat serve',
             'mono': True,
             'size': 12.5},
            {'id': 'loc',
             'x': 172,
             'y': 80,
             'w': 170,
             'h': 64,
             'title': 'find_ovms()',
             'mono': True,
             'sub': ['config, OVAT_OVMS,', 'PATH, known dirs']},
            {'id': 'fd', 'x': 380, 'y': 80, 'w': 130, 'h': 64, 'kind': 'q', 'title': 'Found?'},
            {'id': 'env',
             'x': 552,
             'y': 80,
             'w': 170,
             'h': 76,
             'title': 'ovms_env()',
             'mono': True,
             'sub': ['PATH, PYTHONHOME,', 'LD_LIBRARY_PATH']},
            {'id': 'sp',
             'x': 752,
             'y': 80,
             'w': 184,
             'h': 76,
             'title': 'Popen',
             'mono': True,
             'sub': ['logs go to a file']},
            {'id': 'offer',
             'x': 250,
             'y': 200,
             'w': 150,
             'h': 64,
             'kind': 'warn',
             'title': 'Offer ovat setup',
             'size': 13,
             'sub': ['at a terminal']},
            {'id': 'ref',
             'x': 420,
             'y': 200,
             'w': 150,
             'h': 64,
             'kind': 'fail',
             'title': 'Refuse',
             'sub': ['no terminal,', 'download nothing']},
            {'id': 'wt',
             'x': 752,
             'y': 196,
             'w': 184,
             'h': 84,
             'title': 'wait_until_ready()',
             'mono': True,
             'size': 12,
             'sub': ['stall budget,', 'no deadline']},
            {'id': 'stall',
             'x': 360,
             'y': 340,
             'w': 180,
             'h': 64,
             'kind': 'warn',
             'title': 'No progress, 300 s',
             'size': 13,
             'sub': ['says it is still running']},
            {'id': 'dead',
             'x': 556,
             'y': 340,
             'w': 180,
             'h': 64,
             'kind': 'fail',
             'title': 'Process exited',
             'sub': ['report, point at the log']},
            {'id': 'rdy',
             'x': 752,
             'y': 340,
             'w': 184,
             'h': 64,
             'kind': 'ok',
             'title': 'Ready',
             'sub': ['health 200, pid saved']}],
  'edges': [{'id': 'a', 'from': 'sv', 'to': 'loc'},
            {'id': 'b', 'from': 'loc', 'to': 'fd'},
            {'id': 'c', 'from': 'fd', 'to': 'offer', 'fs': 'b', 'ts': 't', 'ft': 0.3, 'my': 172},
            {'id': 'd', 'from': 'fd', 'to': 'ref', 'fs': 'b', 'ts': 't', 'ft': 0.7, 'my': 172},
            {'id': 'e', 'from': 'fd', 'to': 'env', 'label': 'yes', 'lx': 531, 'ly': 112},
            {'id': 'f', 'from': 'env', 'to': 'sp'},
            {'id': 'g', 'from': 'sp', 'to': 'wt', 'fs': 'b', 'ts': 't'},
            {'id': 'h', 'from': 'wt', 'to': 'rdy', 'fs': 'b', 'ts': 't'},
            {'id': 'i', 'pts': [[752, 262], [646, 262], [646, 340]]},
            {'id': 'j', 'pts': [[752, 226], [600, 226], [600, 316], [450, 316], [450, 340]]}],
  'steps': [{'n': ['sv'], 'c': 'ovat serve starts the OVMS server.', 'd': 2.2},
            {'e': ['a'],
             'n': ['loc'],
             'c': 'find_ovms() looks in the config, OVAT_OVMS, PATH and known folders.'},
            {'e': ['b'], 'n': ['fd'], 'c': 'Was it found?', 'd': 2},
            {'e': ['c'], 'n': ['offer'], 'c': 'No, at a terminal: it offers to run ovat setup.'},
            {'e': ['d'],
             'n': ['ref'],
             'c': 'No, and no terminal: it refuses and downloads nothing.'},
            {'e': ['e'],
             'n': ['env'],
             'c': 'Yes: ovms_env() sets PATH, PYTHONHOME and LD_LIBRARY_PATH.'},
            {'e': ['f'], 'n': ['sp'], 'c': 'Popen starts OVMS, with its logs going to a file.'},
            {'e': ['g'],
             'n': ['wt'],
             'c': 'wait_until_ready() watches for progress. There is no fixed deadline.'},
            {'e': ['h'], 'n': ['rdy'], 'c': 'Health returns 200: ready. The pid goes in ovms.pid.'},
            {'e': ['i'],
             'n': ['dead'],
             'c': 'The process exited: OVAT reports it and points at the log.'},
            {'e': ['j'],
             'n': ['stall'],
             'c': 'No progress for 300 s: it says the server is still running.',
             'd': 3.6}],
  'name': 'serve'},
 {'id': 'device-routing',
  'w': 880,
  'h': 484,
  'group': 'arch',
  'kicker': 'device routing as a decision',
  'title': 'Device routing as a decision',
  'desc': 'At startup OVAT checks the devices. CPU only: everything on the CPU. CPU and GPU: the '
          'LLM on the GPU, embeddings and whisper on the CPU. CPU, GPU and NPU: the LLM on the '
          'GPU, embeddings on the NPU, whisper on the CPU. Serving an LLM on the NPU needs an '
          'int4-cw-ov export, a capped prompt, no batching or beam search, and ignores KV cache '
          'settings.',
  'summary': 'CPU only, CPU + GPU, or CPU + GPU + NPU. An LLM on the NPU comes with four limits.',
  'nodes': [{'id': 'st',
             'x': 24,
             'y': 150,
             'w': 140,
             'h': 40,
             'pill': True,
             'title': 'ovat startup',
             'size': 13},
            {'id': 'det', 'x': 200, 'y': 142, 'w': 130, 'h': 56, 'kind': 'q', 'title': 'Devices?'},
            {'id': 'c1',
             'x': 380,
             'y': 64,
             'w': 220,
             'h': 60,
             'title': 'CPU only',
             'sub': ['everything on the CPU']},
            {'id': 'c2',
             'x': 380,
             'y': 140,
             'w': 220,
             'h': 76,
             'title': 'CPU + GPU',
             'sub': ['LLM: GPU', 'embeddings, whisper: CPU']},
            {'id': 'c3',
             'x': 380,
             'y': 232,
             'w': 220,
             'h': 76,
             'title': 'CPU + GPU + NPU',
             'sub': ['LLM: GPU, embeddings: NPU', 'whisper: CPU']},
            {'id': 'lim',
             'x': 640,
             'y': 240,
             'w': 220,
             'h': 60,
             'title': 'LLM on the NPU',
             'sub': ['four limits']},
            {'id': 'l1',
             'x': 24,
             'y': 350,
             'w': 196,
             'h': 56,
             'kind': 'warn',
             'title': 'Export',
             'sub': [{'c': 'needs -int4-cw-ov'}]},
            {'id': 'l2',
             'x': 234,
             'y': 350,
             'w': 196,
             'h': 56,
             'kind': 'warn',
             'title': 'Prompt capped',
             'sub': [{'c': '--max_prompt_len'}]},
            {'id': 'l3',
             'x': 444,
             'y': 350,
             'w': 196,
             'h': 56,
             'kind': 'warn',
             'title': 'No batching',
             'sub': ['or beam search']},
            {'id': 'l4',
             'x': 654,
             'y': 350,
             'w': 206,
             'h': 56,
             'kind': 'warn',
             'title': 'KV cache',
             'sub': ['settings ignored']}],
  'edges': [{'id': 'a', 'from': 'st', 'to': 'det'},
            {'id': 'b1', 'from': 'det', 'to': 'c1', 'ft': 0.3, 'mx': 355},
            {'id': 'b2', 'from': 'det', 'to': 'c2'},
            {'id': 'b3', 'from': 'det', 'to': 'c3', 'ft': 0.7, 'mx': 355},
            {'id': 'c', 'from': 'c3', 'to': 'lim', 'tt': 0.63},
            {'id': 'd1', 'from': 'lim', 'to': 'l1', 'fs': 'b', 'ts': 't', 'my': 328},
            {'id': 'd2', 'from': 'lim', 'to': 'l2', 'fs': 'b', 'ts': 't', 'my': 328},
            {'id': 'd3', 'from': 'lim', 'to': 'l3', 'fs': 'b', 'ts': 't', 'my': 328},
            {'id': 'd4', 'from': 'lim', 'to': 'l4', 'fs': 'b', 'ts': 't', 'my': 328}],
  'steps': [{'e': ['a'],
             'n': ['st', 'det'],
             'c': 'At startup OVAT checks which Intel devices the machine has.'},
            {'e': ['b1'],
             'n': ['c1'],
             'c': 'CPU only, like a Mac or a dev laptop: everything runs on the CPU.'},
            {'e': ['b2'],
             'n': ['c2'],
             'c': 'CPU + GPU: the LLM on the GPU, embeddings and whisper on the CPU.'},
            {'e': ['b3'],
             'n': ['c3'],
             'c': 'CPU + GPU + NPU: LLM on the GPU, embeddings on the NPU, whisper on the CPU.'},
            {'e': ['c'], 'n': ['lim'], 'c': 'Serving the LLM itself on the NPU comes with limits:'},
            {'e': ['d1', 'd2', 'd3', 'd4'],
             'n': ['l1', 'l2', 'l3', 'l4'],
             'c': 'an int4-cw-ov export, a capped prompt, no batching or beam search, no KV cache '
                  'settings.',
             'd': 4}],
  'name': 'device-routing'},
 {'id': 'model-scout',
  'w': 880,
  'h': 404,
  'group': 'arch',
  'kicker': 'how a model folder is classified',
  'title': 'How a model folder is classified',
  'desc': 'If a folder has openvino_vision or openvino_language_model files and its model_type is '
          'a unified type (qwen3_5, qwen3_5_text, qwen3_6), its kind is unified and one '
          'VLMPipeline answers both filters. Other vision models (qwen2_vl, internvl_chat, phi3_v) '
          'are vlm. Folders without those files go to the other checks: whisper, embeddings, or an '
          'LLM.',
  'summary': 'Vision files and a unified type: unified. Vision files only: vlm. Neither: the other '
             'checks.',
  'nodes': [{'id': 'f',
             'x': 24,
             'y': 104,
             'w': 150,
             'h': 40,
             'pill': True,
             'title': 'model folder',
             'mono': True,
             'size': 12.5},
            {'id': 'c',
             'x': 204,
             'y': 80,
             'w': 250,
             'h': 88,
             'kind': 'q',
             'title': 'Vision files?',
             'sub': [{'c': 'openvino_vision_*.xml'}, {'c': 'openvino_language_model.xml'}]},
            {'id': 'l',
             'x': 204,
             'y': 220,
             'w': 250,
             'h': 84,
             'title': 'Other checks',
             'sub': ['whisper, embeddings,', 'or llm: LLMPipeline']},
            {'id': 'm',
             'x': 500,
             'y': 80,
             'w': 160,
             'h': 88,
             'kind': 'q',
             'title': 'Unified type?',
             'sub': [{'c': 'model_type in'}, {'c': '_UNIFIED_TYPES'}]},
            {'id': 'v',
             'x': 500,
             'y': 220,
             'w': 160,
             'h': 100,
             'title': 'kind = vlm',
             'mono': True,
             'sub': ['vision only', {'c': 'qwen2_vl, phi3_v'}, {'c': 'internvl_chat'}]},
            {'id': 'u',
             'x': 700,
             'y': 80,
             'w': 160,
             'h': 130,
             'kind': 'ok',
             'title': 'kind = unified',
             'mono': True,
             'size': 12.5,
             'sub': ['one VLMPipeline,',
                     'both filters',
                     {'c': 'qwen3_5, qwen3_6'},
                     {'c': 'qwen3_5_text'}]}],
  'edges': [{'id': 'a', 'from': 'f', 'to': 'c'},
            {'id': 'b', 'from': 'c', 'to': 'l', 'fs': 'b', 'ts': 't', 'label': 'no'},
            {'id': 'd', 'from': 'c', 'to': 'm', 'label': 'yes'},
            {'id': 'e', 'from': 'm', 'to': 'v', 'fs': 'b', 'ts': 't', 'label': 'no'},
            {'id': 'g', 'from': 'm', 'to': 'u', 'tt': 0.34, 'label': 'yes', 'lx': 680, 'ly': 124}],
  'steps': [{'n': ['f'],
             'c': 'OVAT looks inside a model folder to decide what kind of model it is.'},
            {'e': ['a'], 'n': ['c'], 'c': 'Are there OpenVINO vision or language-model files?'},
            {'e': ['b'],
             'n': ['l'],
             'c': 'No: the other checks run: whisper, embeddings, or an LLM for LLMPipeline.'},
            {'e': ['d'], 'n': ['m'], 'c': 'Yes: is its model_type one of the unified types?'},
            {'e': ['e'],
             'n': ['v'],
             'c': 'No, like qwen2_vl, internvl_chat or phi3_v: kind is vlm, vision only.'},
            {'e': ['g'],
             'n': ['u'],
             'c': 'Yes, like qwen3_5 or qwen3_6: kind is unified. One VLMPipeline answers both '
                  'filters.',
             'd': 4}],
  'name': 'model-scout'},
 {'id': 'reply-check',
  'w': 880,
  'h': 472,
  'group': 'arch',
  'kicker': 'a reply with no tool calls',
  'title': 'A reply with no tool calls',
  'desc': 'If tool markup is still in the text and finish_reason is length, the reply was cut at '
          'max_tokens and the markup is a truncated fragment. Markup without length is '
          'undecoded_tool_call: the server did not decode it. No markup and nothing after the '
          'reasoning is empty_answer. Otherwise it is a real answer.',
  'summary': 'Markup and length: truncated. Markup alone: undecoded_tool_call. Nothing left: '
             'empty_answer. Else the answer.',
  'nodes': [{'id': 'r',
             'x': 24,
             'y': 96,
             'w': 150,
             'h': 40,
             'pill': True,
             'title': 'no tool_calls',
             'mono': True,
             'size': 12.5},
            {'id': 'q1',
             'x': 200,
             'y': 80,
             'w': 190,
             'h': 72,
             'kind': 'q',
             'title': 'Markup left?',
             'sub': ['tool markup still', 'in the text']},
            {'id': 'q0',
             'x': 440,
             'y': 80,
             'w': 200,
             'h': 72,
             'kind': 'q',
             'title': 'Cut at length?',
             'sub': [{'c': 'finish_reason'}]},
            {'id': 't',
             'x': 680,
             'y': 72,
             'w': 180,
             'h': 88,
             'kind': 'fail',
             'title': 'Truncated',
             'sub': ['cut at max_tokens,', 'markup is a fragment']},
            {'id': 'a',
             'x': 680,
             'y': 200,
             'w': 180,
             'h': 84,
             'kind': 'fail',
             'title': 'undecoded_tool_call',
             'mono': True,
             'size': 11.5,
             'sub': ['the server did not', 'decode the markup']},
            {'id': 'q2',
             'x': 200,
             'y': 200,
             'w': 190,
             'h': 72,
             'kind': 'q',
             'title': 'Answer left?',
             'sub': ['anything after', 'the reasoning?']},
            {'id': 'c',
             'x': 440,
             'y': 204,
             'w': 200,
             'h': 64,
             'kind': 'ok',
             'title': 'A real answer'},
            {'id': 'b',
             'x': 200,
             'y': 320,
             'w': 190,
             'h': 72,
             'kind': 'fail',
             'title': 'empty_answer',
             'mono': True,
             'sub': ['the parser swallowed', 'the reply']}],
  'edges': [{'id': 'e1', 'from': 'r', 'to': 'q1'},
            {'id': 'e2', 'from': 'q1', 'to': 'q0', 'label': 'yes'},
            {'id': 'e3', 'from': 'q0', 'to': 't', 'label': 'yes'},
            {'id': 'e4',
             'from': 'q0',
             'to': 'a',
             'fs': 'b',
             'ts': 't',
             'my': 178,
             'label': 'no',
             'lx': 610,
             'ly': 178},
            {'id': 'e5', 'from': 'q1', 'to': 'q2', 'fs': 'b', 'ts': 't', 'label': 'no'},
            {'id': 'e6', 'from': 'q2', 'to': 'b', 'fs': 'b', 'ts': 't', 'label': 'no'},
            {'id': 'e7', 'from': 'q2', 'to': 'c', 'label': 'yes'}],
  'steps': [{'n': ['r'], 'c': 'A reply came back with no tool_calls. What does it mean?'},
            {'e': ['e1'], 'n': ['q1'], 'c': 'Is there tool markup still in the text?'},
            {'e': ['e2'], 'n': ['q0'], 'c': 'Yes. Did it stop because finish_reason is length?'},
            {'e': ['e3'],
             'n': ['t'],
             'c': 'Yes: it was cut at max_tokens. The markup is a truncated fragment.'},
            {'e': ['e4'],
             'n': ['a'],
             'c': 'No: the server did not decode the markup. That is undecoded_tool_call.'},
            {'e': ['e5'], 'n': ['q2'], 'c': 'No markup. Is anything left after the reasoning?'},
            {'e': ['e6'],
             'n': ['b'],
             'c': 'No: the parser swallowed the reply. That is empty_answer.'},
            {'e': ['e7'], 'n': ['c'], 'c': 'Yes: it is a real answer.', 'd': 3.4}],
  'name': 'reply-check'},
 {'id': 'plano',
  'w': 880,
  'h': 360,
  'group': 'arch',
  'kicker': 'the plano request path',
  'title': 'The plano request path',
  'desc': 'ovat run points ovms_url at :8000/v1, where plano (Envoy) listens and records '
          'OpenTelemetry spans. Requests pass through ovms_id_bridge.py on :8001 to OVMS on '
          ':8002/v3, the port in model.ovms_port. planoai obs shows latency, time to first token '
          'and tokens.',
  'summary': 'ovat to plano :8000 to the bridge :8001 to OVMS :8002. planoai obs reads the spans.',
  'nodes': [{'id': 'o',
             'x': 24,
             'y': 90,
             'w': 170,
             'h': 80,
             'title': 'ovat run',
             'mono': True,
             'sub': [{'c': 'ovms_url :8000/v1'}]},
            {'id': 'p',
             'x': 240,
             'y': 90,
             'w': 180,
             'h': 80,
             'title': 'plano (Envoy)',
             'sub': [':8000, OTel spans']},
            {'id': 'b',
             'x': 466,
             'y': 90,
             'w': 180,
             'h': 80,
             'title': 'ovms_id_bridge.py',
             'mono': True,
             'size': 12.5,
             'sub': [':8001']},
            {'id': 'v',
             'x': 692,
             'y': 90,
             'w': 168,
             'h': 80,
             'kind': 'ok',
             'title': 'OVMS',
             'sub': [{'c': ':8002/v3'}, 'model.ovms_port']},
            {'id': 'obs',
             'x': 240,
             'y': 214,
             'w': 180,
             'h': 66,
             'optional': True,
             'title': 'planoai obs',
             'mono': True,
             'sub': ['latency, TTFT, tokens']}],
  'edges': [{'id': 'a', 'from': 'o', 'to': 'p'},
            {'id': 'c', 'from': 'p', 'to': 'b'},
            {'id': 'd', 'from': 'b', 'to': 'v'},
            {'id': 'e', 'from': 'p', 'to': 'obs', 'fs': 'b', 'ts': 't', 'dashed': True}],
  'steps': [{'n': ['o'], 'c': 'ovat run points ovms_url at :8000/v1, where plano listens.'},
            {'e': ['a'],
             'n': ['p'],
             'c': 'plano, built on Envoy, records an OpenTelemetry span for each request.'},
            {'e': ['c'], 'n': ['b'], 'c': 'ovms_id_bridge.py on :8001 passes the request on.'},
            {'e': ['d'],
             'n': ['v'],
             'c': 'OVMS answers on :8002/v3, the port set in model.ovms_port.'},
            {'e': ['e'],
             'n': ['obs'],
             'c': 'planoai obs shows latency, time to first token and tokens.',
             'd': 3.6}],
  'name': 'plano'}]

DIAGRAMS = SPECS_README + SPECS_ARCH


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--site-out", type=Path, help="write the site figures here, one HTML file per diagram")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in DIAGRAMS:
        for theme in ("light", "dark"):
            path = args.out / f"{spec['id']}-{theme}.svg"
            path.write_text(build_svg(spec, theme), encoding="utf-8")
            print("wrote", path.name)
        if args.site_out:
            args.site_out.mkdir(parents=True, exist_ok=True)
            (args.site_out / f"{spec['id']}.html").write_text(build_site(spec) + "\n", encoding="utf-8")
            print("wrote site", spec["id"])


if __name__ == "__main__":
    main()
