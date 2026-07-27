"""Packaging metadata that keeps the contributor environment reproducible."""

from pathlib import Path
import tomllib


def test_dev_extra_includes_every_optional_agent_framework():
    """A fresh ``.[dev]`` install must run tests for every agent engine."""
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]

    dev_dependencies = set(extras["dev"])
    for framework_extra in ("langchain", "llamaindex", "openai-agents"):
        assert set(extras[framework_extra]) <= dev_dependencies
