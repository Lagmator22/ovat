"""Apache-2.0 section 4(b): every copied file OVAT changed says so."""
import pathlib

VIBE = pathlib.Path(__file__).resolve().parents[1] / "ovat" / "tui" / "vibe"
NOTE = "Changed by OVAT from Mistral Vibe (commit 4ae5c59). See NOTICE."


def test_the_changed_copied_files_carry_a_notice():
    """65 Python files and one stylesheet differ from upstream at 4ae5c59
    beyond the package rename (counted 2026-11-18). Only 17 of them said
    so. A file losing its first line is caught here; a NEWLY changed file
    still needs the line added by hand when it is first edited."""
    marked = [p for p in VIBE.rglob("*") if p.suffix in (".py", ".tcss")
              and "__pycache__" not in p.parts
              and NOTE in p.read_text(encoding="utf-8").split("\n", 1)[0]]
    assert len(marked) >= 66, f"only {len(marked)} files carry the notice"
