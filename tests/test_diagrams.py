"""The animated diagrams and the images the docs point at.

All 21 animated SVG diagram pairs under docs/assets/diagrams/ are written by
scripts/gen_agent_diagrams.py from one spec each. If someone edits an SVG by
hand, or edits the spec and forgets to run the script, the committed file no
longer says what the spec says; this test fails then.
"""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("gen_agent_diagrams", ROOT / "scripts" / "gen_agent_diagrams.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


@pytest.mark.parametrize("diagram", gen.DIAGRAMS, ids=lambda d: d["id"])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_committed_svg_is_what_the_script_writes(diagram, theme):
    path = ROOT / "docs" / "assets" / "diagrams" / f"{diagram['id']}-{theme}.svg"
    assert path.read_text(encoding="utf-8") == gen.build_svg(diagram, theme), (
        f"{path.name} is stale: run python scripts/gen_agent_diagrams.py")


@pytest.mark.parametrize("diagram", gen.DIAGRAMS, ids=lambda d: d["id"])
def test_every_step_names_cards_and_a_path_that_exist(diagram):
    nodes = {n["id"] for n in diagram.get("nodes") or []}
    edges = {e["id"] for e in diagram.get("edges") or []}
    for step in diagram.get("steps") or []:
        assert set(step.get("n") or []) <= nodes
        assert set(step.get("e") or []) <= edges


@pytest.mark.parametrize("diagram", gen.DIAGRAMS, ids=lambda d: d["id"])
def test_the_site_version_has_one_list_item_per_step(diagram):
    html = gen.build_site(diagram)
    steps = diagram.get("steps") or []
    assert html.count("<li ") == len(steps)
    assert f"1 / {len(steps)}" in html


def test_a_line_too_wide_for_its_card_is_refused():
    bad = dict(gen.DIAGRAMS[0])
    bad["nodes"] = [{
        "id": "x",
        "x": 0,
        "y": 0,
        "w": 100,
        "h": 100,
        "title": "t",
        "sub": ["this line is far too wide for a card"],
    }]
    with pytest.raises(AssertionError, match="too wide"):
        gen.check_fit(bad)


def _docs_with_images():
    yield ROOT / "README.md"
    yield from sorted((ROOT / "docs").glob("*.md"))
    yield from sorted((ROOT / "examples").glob("*/README.md"))


@pytest.mark.parametrize("doc", list(_docs_with_images()), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_local_image_a_doc_names_exists(doc):
    text = doc.read_text(encoding="utf-8")
    missing = []
    for ref in re.findall(r'(?:src|srcset)="([^"]+)"', text):
        if ref.startswith(("http://", "https://", "data:")):
            continue
        if not (doc.parent / ref).exists():
            missing.append(ref)
    assert not missing, f"{doc.name} points at files that are not there: {missing}"
