# tests/test_docs.py
"""The workflow reference must describe the schema that actually ships.

A reference table is a second copy of the schema, and this project's rule is
that two copies drift. These are the two ways it would: a field added without
a description, and a default changed without the table being regenerated. Both
produce a document that is confidently wrong, which is worse than no document
because a reader has no reason to distrust it.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "gen_workflow_reference.py")
DOC = os.path.join(ROOT, "docs", "workflow_yaml_reference.md")


def _generator():
    """The generator module, imported without running it."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        import gen_workflow_reference
        return gen_workflow_reference
    finally:
        sys.path.pop(0)


def test_every_schema_field_has_a_description():
    """Add a field, forget the doc, and this names it.

    The failure mode without this is silent: the table still generates, the
    new row simply has an empty Purpose cell, and nobody notices until a user
    asks what the field does.
    """
    generator = _generator()

    undocumented = []
    for title, model, _ in generator.SECTIONS:
        for name in model.model_fields:
            if not generator.PURPOSE.get((title, name)):
                undocumented.append(f"{title}.{name}")
    assert not undocumented, (
        "these fields have no line in PURPOSE in "
        "scripts/gen_workflow_reference.py:\n  " + "\n  ".join(undocumented))


def test_the_committed_reference_is_up_to_date():
    """Change a default, forget to regenerate, and this fails.

    Run `python scripts/gen_workflow_reference.py` to fix.
    """
    result = subprocess.run(
        [sys.executable, SCRIPT, "--check"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        result.stdout + result.stderr +
        "\n\nRun: python scripts/gen_workflow_reference.py")


def test_the_reference_covers_every_config_class():
    """A new config SECTION must be added to SECTIONS, not just new fields.

    Catches the other half: someone adds a whole `vlm:` block and documents
    none of it, which the per-field check above cannot see because the class
    is not in SECTIONS at all.
    """
    from ovat.config import workflow

    generator = _generator()
    documented = {model for _, model, _ in generator.SECTIONS}
    missing = []
    for name in dir(workflow):
        value = getattr(workflow, name)
        if (isinstance(value, type)
                and issubclass(value, workflow.StrictModel)
                and value is not workflow.StrictModel
                and value not in documented
                # RagConfig holds only the three sub-sections, each of which
                # is documented in its own table.
                and value is not workflow.RagConfig):
            missing.append(name)
    assert not missing, (
        "config classes missing from SECTIONS in the generator: "
        f"{missing}")


def test_the_example_in_the_reference_actually_validates(tmp_path):
    """The YAML at the bottom is what a reader copies first.

    A reference whose own example fails validation is the worst possible
    version of this document.
    """
    import re

    import yaml

    from ovat.config.workflow import WorkflowConfig

    with open(DOC, encoding="utf-8") as handle:
        text = handle.read()
    blocks = re.findall(r"```yaml\n(.*?)```", text, re.S)
    assert blocks, "no yaml example found in the reference"

    for block in blocks:
        data = yaml.safe_load(block)
        WorkflowConfig(**data)          # raises if the example is wrong


def test_the_generator_is_not_shipped_in_the_wheel():
    """scripts/ is a development tool, not part of the package.

    pyproject includes only `ovat*`, and this pins that: a generator that
    landed inside the package would be importable by users for no reason and
    would drag its assumptions about repo layout with it.
    """
    import tomllib

    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
        data = tomllib.load(handle)
    include = data["tool"]["setuptools"]["packages"]["find"]["include"]
    assert include == ["ovat*"], include
    assert not os.path.exists(os.path.join(ROOT, "ovat", "scripts"))
