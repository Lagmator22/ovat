"""Packaging metadata that keeps the contributor environment reproducible."""

from pathlib import Path
import tomllib


def _extras():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return metadata["project"]["optional-dependencies"]


def test_an_extra_installs_the_converter_the_docs_tell_users_to_run():
    """README and `ovat init` both instruct the user to run

        optimum-cli export openvino --model BAAI/bge-small-en-v1.5 ...

    to enable RAG. Nothing installed optimum-cli, so that line was a
    guaranteed "command not found" on a clean machine: it is not a
    dependency, and a full `.[langchain,llamaindex,openai-agents,tui]`
    install does not provide it. There is no pre-converted bge-small in the
    OpenVINO org either, so conversion is the only route to real retrieval.

    Deliberately an extra, not a core dependency: it pulls torch, transformers
    and nncf, which is a multi-GB install nobody running `ovat run` needs.
    """
    extras = _extras()
    assert "convert" in extras, "no extra provides optimum-cli"
    assert any("optimum-intel" in dep for dep in extras["convert"])
    # The openvino extra is what registers the `export openvino` target.
    assert any("openvino" in dep for dep in extras["convert"])


def test_the_model_download_tool_is_a_core_dependency():
    """The README tells users to fetch an IR model with `hf download`.

    That has to work on a plain install. `ovms --pull` is the other route,
    but OVMS does not run on macOS at all, so without this there is no
    documented way to get a model onto a dev machine -- and `ovat chat`,
    the whole macOS story, needs one on disk.

    Checked as a declared dependency rather than by importing, so the test
    still means something when run inside an environment that happens to
    have huggingface_hub installed for some other reason.
    """
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps = metadata["project"]["dependencies"]
    assert any("huggingface" in d.replace("_", "-").lower() for d in deps)


def test_dev_extra_includes_every_optional_agent_framework():
    """A fresh ``.[dev]`` install must run tests for every agent engine."""
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]

    dev_dependencies = set(extras["dev"])
    for framework_extra in ("langchain", "llamaindex", "openai-agents"):
        assert set(extras[framework_extra]) <= dev_dependencies
