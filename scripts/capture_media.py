#!/usr/bin/env python3
"""Turn real OVAT command output into the SVG screenshots used in the docs.

Two modes, because the interesting commands do not all run on one machine:

    python scripts/capture_media.py run        # run commands HERE and capture
    python scripts/capture_media.py convert DIR  # convert transcripts from the AI PC

`run` executes each command as a real subprocess with colour forced on, so the
image is a recording of the actual program, never a hand-typed mock-up. It only
covers commands that need no model server. `serve`, `run` and `bench` do need
one, so they are captured on the AI PC by aipc/ovat-demo.cmd, which writes the
same kind of ANSI transcript that `convert` reads back into SVGs.

Why a custom code_format: rich's default SVG template pulls Fira Code from
cdnjs. An SVG loaded through an <img> tag -- which is how GitHub renders one --
cannot fetch external resources at all, so that request is dead weight, and it
makes the file non-reproducible offline. The template below asks for locally
installed monospace faces instead and is otherwise rich's own.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "assets"
WIDTH = 100

# rich's default template with the @font-face block replaced by a local stack.
CODE_FORMAT = """<svg class="rich-terminal" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
    <style>
    .{unique_id}-matrix {{
        font-family: "SFMono-Regular", "Menlo", "DejaVu Sans Mono", "Consolas", monospace;
        font-size: {char_height}px;
        line-height: {line_height}px;
        font-variant-east-asian: full-width;
    }}
    .{unique_id}-title {{
        font-size: 18px;
        font-weight: bold;
        font-family: "SFMono-Regular", "Menlo", monospace;
    }}
    {styles}
    </style>
    <defs>
    <clipPath id="{unique_id}-clip-terminal">
      <rect x="0" y="0" width="{terminal_width}" height="{terminal_height}" />
    </clipPath>
    {lines}
    </defs>
    {chrome}
    <g transform="translate({terminal_x}, {terminal_y})" clip-path="url(#{unique_id}-clip-terminal)">
    {backgrounds}
    <g class="{unique_id}-matrix">
    {matrix}
    </g>
    </g>
</svg>
"""

# (slug, title, argv). Only commands that need no model server live here.
# `doctor` is the only local command whose output is worth a picture: it is a
# full table that differs per machine. `run --dry-run` prints three lines, and
# serve / run / bench need OVMS, so those are captured on the AI PC instead and
# come back through `convert`.
LOCAL_COMMANDS = [
    ("doctor", "ovat doctor", ["doctor"]),
]


def _ovat() -> str:
    """The venv's console script if there is one, else whatever is on PATH."""
    for candidate in (REPO / ".venv" / "bin" / "ovat",
                      REPO / ".venv" / "Scripts" / "ovat.exe"):
        if candidate.exists():
            return str(candidate)
    return "ovat"


def _env() -> dict:
    """Force colour and a fixed width, so captures are reproducible."""
    return dict(os.environ, FORCE_COLOR="1", TERM="xterm-256color",
                COLUMNS=str(WIDTH), PYTHONIOENCODING="utf-8", OVAT_TUI="")


def ansi_to_svg(raw: str, title: str, dest: Path) -> None:
    from rich.console import Console
    from rich.text import Text

    console = Console(record=True, width=WIDTH, file=open(os.devnull, "w"),
                      force_terminal=True, color_system="truecolor")
    console.print(Text.from_ansi(raw.rstrip("\n")))
    dest.write_text(console.export_svg(title=title, code_format=CODE_FORMAT))
    print(f"  wrote {dest.relative_to(REPO)}  ({dest.stat().st_size // 1024} KB)")


def do_run() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ovat = _ovat()
    for slug, title, argv in LOCAL_COMMANDS:
        print(f"capturing: {title}")
        try:
            proc = subprocess.run([ovat, *argv], capture_output=True, text=True,
                                  env=_env(), cwd=REPO, timeout=600)
        except FileNotFoundError:
            print(f"  SKIP: {ovat} not found. pip install -e '.[dev]' first.")
            continue
        except subprocess.TimeoutExpired:
            print("  SKIP: timed out.")
            continue
        ansi_to_svg(proc.stdout + proc.stderr, title, OUT / f"ovat-{slug}.svg")
    return 0


def do_convert(folder: str) -> int:
    """Convert an AI PC capture folder (ovat-demo.cmd capture) into SVGs."""
    src = Path(folder)
    if not src.is_dir():
        print(f"not a directory: {src}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    titles = {
        "doctor": "ovat doctor",
        "bench": "ovat bench  (all four engines)",
        "run-native": "ovat run  (native loop)",
        "run-react": "ovat run --react  (LangChain)",
        "run-llamaindex": "ovat run --llamaindex",
        "run-openai-agents": "ovat run --openai-sdk",
        "telemetry": "ovat telemetry",
    }
    found = 0
    for txt in sorted(src.glob("*.txt")):
        slug = txt.stem
        raw = txt.read_text(encoding="utf-8", errors="replace")
        ansi_to_svg(raw, titles.get(slug, f"ovat {slug}"), OUT / f"ovat-{slug}.svg")
        found += 1
    if not found:
        print(f"no .txt transcripts in {src}")
        return 1
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        raise SystemExit(do_run())
    if mode == "convert":
        if len(sys.argv) < 3:
            print("usage: capture_media.py convert <folder-from-the-AI-PC>")
            raise SystemExit(1)
        raise SystemExit(do_convert(sys.argv[2]))
    print(__doc__)
    raise SystemExit(1)
