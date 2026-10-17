# tests/test_tui_branding.py
"""The TUI must not show Mistral's brand.

ovat/tui/vibe/ is copied from Mistral Vibe (Apache-2.0). The code is theirs,
the words the user reads must be OVAT's. This scans every STRING LITERAL under
ovat/tui/ (with `ast`, so comments are not looked at: the Apache licence
needs the copyright and origin comments to stay) and fails when one carries a
Mistral brand word.

Code identifiers (class and function names such as `MistralColors`, CSS
variable names) are not strings and are allowed on purpose: the copies stay
diffable with upstream.
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TUI = ROOT / "ovat" / "tui"

FORBIDDEN = re.compile(r"mistral|le chat|mistral\.ai|leanstral|chonk", re.IGNORECASE)

# (path relative to the repo, the exact string) -> why it is allowed.
ALLOWED: dict[tuple[str, str], str] = {
    ("ovat/tui/vibe/config_values.py", "mistral"):
        "AudioClient is a wire value of the copied config models "
        "(AudioProviderView.client). It is validated, never displayed.",
    ("ovat/tui/vibe/app_server/protocol.py", "mistral"):
        "default of CompletionConfig.type in a copied wire model; never "
        "displayed, and OVAT sends its own value.",
    ("ovat/tui/vibe/app_server/protocol.py", "mistral-small-latest"):
        "default of CompletionConfig.model in the same wire model.",
}
# Strings that START with an allowed prefix: the origin notes the Apache
# licence asks for, in docstrings (comments are not scanned at all).
ALLOWED_PREFIX: dict[tuple[str, str], str] = {
    ("ovat/tui/__init__.py", "The agent TUI (`ovat agent`)"):
        "module docstring that credits Mistral Vibe as the design reference "
        "and the source of the copied widgets, as Apache-2.0 requires.",
}


def _literals(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def _allowed(rel: str, text: str) -> bool:
    if (rel, text) in ALLOWED:
        return True
    return any(rel == r and text.startswith(prefix)
               for (r, prefix) in ALLOWED_PREFIX)


def test_no_mistral_brand_in_any_string_under_ovat_tui():
    hits = []
    for path in sorted(TUI.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        for line, text in _literals(path):
            if FORBIDDEN.search(text) and not _allowed(rel, text):
                hits.append(f"{rel}:{line}: {text[:80]!r}")
    assert not hits, "Mistral brand text in OVAT's TUI:\n" + "\n".join(hits)


def test_the_scanner_catches_a_brand_word(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text('# a comment about Mistral is fine\nX = "Welcome to Le Chat"\n')
    found = [t for _, t in _literals(sample) if FORBIDDEN.search(t)]
    assert found == ["Welcome to Le Chat"]


def test_every_allowed_entry_is_still_needed():
    # A stale allowlist hides the next real hit.
    live = set()
    for path in TUI.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        for _, text in _literals(path):
            if FORBIDDEN.search(text):
                live.add((rel, text))
    for key in ALLOWED:
        assert key in live, f"stale allowlist entry: {key}"
    for rel, prefix in ALLOWED_PREFIX:
        assert any(r == rel and t.startswith(prefix) for r, t in live), (
            f"stale allowlist entry: {(rel, prefix)}")


def test_the_cat_mascot_is_gone():
    banner = TUI / "vibe" / "cli" / "textual_ui" / "widgets" / "banner"
    assert not (banner / "petit_chat.py").exists()
    assert "PetitChat" not in (banner / "banner.py").read_text(encoding="utf-8")
    assert "OVAT" in (banner / "banner.py").read_text(encoding="utf-8")
