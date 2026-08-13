"""Packaging metadata that keeps the contributor environment reproducible."""

from pathlib import Path

# tomllib is stdlib only from 3.11. The project supports 3.10, and the README
# advertises it, so importing it bare made this whole module fail to collect
# on the oldest Python we claim to support -- a packaging test that cannot run
# on a supported interpreter is worse than none. tomli is the same library
# under its pre-stdlib name, and rides in the dev extra for 3.10 only.
try:
    import tomllib
except ModuleNotFoundError:                  # Python 3.10
    import tomli as tomllib


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


def test_no_prose_is_trapped_inside_a_shell_code_fence():
    """A copy button that pastes English into a terminal.

    examples/plano/README.md had five lines of explanation inside a ```cmd
    fence, so anyone clicking Copy pasted prose at their shell. Checked for
    every doc rather than the one that had it, since the mistake is invisible
    in a rendered preview.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    shell = {"cmd", "bash", "console", "bat", "sh", "powershell"}
    offenders = []
    for path in list(root.glob("*.md")) + list(root.glob("docs/*.md")) \
            + list(root.glob("examples/**/*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        language, start = None, 0
        for number, line in enumerate(lines, 1):
            fence = re.match(r"^```(\w*)", line.strip())
            if not fence:
                continue
            if language is None:
                language, start = fence.group(1).lower(), number
                continue
            if language in shell:
                for offset, body in enumerate(lines[start:number - 1], start + 1):
                    text = body.strip()
                    # A prose line: ends in a full stop and is not a comment
                    # or a line continuation.
                    if (text.endswith(".") and not text.startswith("#")
                            and not text.startswith("::")
                            and not text.endswith("\\")
                            and " " in text and len(text.split()) > 4):
                        offenders.append(f"{path.name}:{offset}: {text[:60]}")
            language = None
    assert not offenders, "prose inside a shell code fence:\n" + "\n".join(offenders)
