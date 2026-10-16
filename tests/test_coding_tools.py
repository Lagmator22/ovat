# tests/test_coding_tools.py
"""The five coding tools: read_file, write_file, edit, grep, shell.

Note to myself: every tool is a plain function taking the project folder
(`root`), so these tests hand it a tmp_path and never touch the real
working directory. The engine-level tests go through build_tools, which is
where the working directory is read.
"""
import os
import sys
import threading
import time

import pytest

from ovat.agent import factory
from ovat.agent.arg_models import args_model_from_schema, json_schema_for_tool
from ovat.agent.events import TOOL_ENDED, EventBus, observed
from ovat.agent.permissions import default_permission
from ovat.config.workflow import WorkflowConfig
from ovat.tools import grep as grep_tool
from ovat.tools.edit import edit_impl
from ovat.tools.grep import grep_impl
from ovat.tools.project import CODING_TOOLS, Shown
from ovat.tools.read_file import read_file_impl
from ovat.tools.shell import shell_impl
from ovat.tools.write_file import write_file_impl

NAMES = ["read_file", "write_file", "edit", "grep", "shell"]


@pytest.fixture
def root(tmp_path):
    return tmp_path.resolve()


def _is_error(text):
    return str(text).startswith("Error:")


# ---- read_file ------------------------------------------------------------

def test_read_file_numbers_the_lines(root):
    (root / "a.txt").write_text("one\ntwo\nthree\n")
    out = read_file_impl("a.txt", root)
    assert "1\u2192one" in out and "3\u2192three" in out
    assert out.display["num_lines"] == 3 and out.display["total_lines"] == 3


def test_read_file_offset_and_limit_say_how_to_continue(root):
    (root / "a.txt").write_text("\n".join(f"l{i}" for i in range(1, 11)))
    out = read_file_impl("a.txt", root, offset=3, limit=2)
    assert "3\u2192l3" in out and "4\u2192l4" in out and "l5" not in out
    assert "offset 5" in out
    assert out.display["was_truncated"] is True


def test_read_file_past_the_end_and_empty_say_so(root):
    (root / "e.txt").write_text("")
    (root / "s.txt").write_text("x\n")
    assert "empty" in read_file_impl("e.txt", root)
    assert "shorter than the offset" in read_file_impl("s.txt", root, offset=9)


def test_read_file_refuses_binary_missing_and_folders(root):
    (root / "b.bin").write_bytes(b"\x00\x01\x02 data")
    (root / "d").mkdir()
    assert "binary" in read_file_impl("b.bin", root)
    assert "not found" in read_file_impl("nope.txt", root)
    assert "folder" in read_file_impl("d", root)
    assert all(_is_error(read_file_impl(p, root))
               for p in ("b.bin", "nope.txt", "d", ""))


def test_read_file_caps_a_huge_file_by_bytes(root):
    (root / "big.txt").write_text("".join("y" * 100 + "\n" for _ in range(2000)))
    out = read_file_impl("big.txt", root)
    assert len(out.encode()) < 70_000
    assert out.display["was_truncated"] is True and "offset" in out


# ---- the project fence ----------------------------------------------------

@pytest.mark.parametrize("tool", ["read", "write", "edit", "grep"])
def test_paths_outside_the_project_are_refused(root, tmp_path_factory, tool):
    project = root / "proj"
    project.mkdir()
    other = tmp_path_factory.mktemp("elsewhere").resolve()
    (other / "secret.txt").write_text("hush")
    target = str(other / "secret.txt")
    out = {
        "read": lambda: read_file_impl(target, project),
        "write": lambda: write_file_impl(target, "x", project),
        "edit": lambda: edit_impl(target, "hush", "no", project),
        "grep": lambda: grep_impl("hush", project, str(other)),
    }[tool]()
    assert _is_error(out) and "outside the project folder" in out
    assert (other / "secret.txt").read_text() == "hush"


def test_dot_dot_is_refused_and_inside_absolute_paths_work(root):
    project = root / "proj"
    project.mkdir()
    (root / "up.txt").write_text("up")
    (project / "in.txt").write_text("in")
    assert "outside the project" in read_file_impl("../up.txt", project)
    assert "1\u2192in" in read_file_impl(str(project / "in.txt"), project)


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on "
                                            "Windows runners")
def test_a_symlink_that_leaves_the_project_is_refused(root, tmp_path_factory):
    project = root / "proj"
    project.mkdir()
    other = tmp_path_factory.mktemp("elsewhere").resolve()
    (other / "secret.txt").write_text("hush")
    (project / "link").symlink_to(other)
    assert "outside the project" in read_file_impl("link/secret.txt", project)
    assert "outside the project" in write_file_impl("link/new.txt", "x",
                                                    project)
    assert not (other / "new.txt").exists()


def test_a_native_style_path_works_on_this_platform(root):
    # On Windows this is C:\...\a.txt with backslashes, the form a user
    # pastes; elsewhere it is the ordinary absolute path.
    (root / "a.txt").write_text("hi")
    native = str(root / "a.txt")
    if os.name == "nt":
        assert "\\" in native
    assert "1\u2192hi" in read_file_impl(native, root)


# ---- write_file -----------------------------------------------------------

def test_write_file_creates_parents_and_reports_created(root):
    out = write_file_impl("new/deep/a.txt", "hello", root)
    assert (root / "new" / "deep" / "a.txt").read_text() == "hello"
    assert out.startswith("Created") and out.display["file_existed"] is False


def test_write_file_overwrites_and_keeps_the_old_text_for_the_diff(root):
    (root / "a.txt").write_text("old")
    out = write_file_impl("a.txt", "new", root)
    assert (root / "a.txt").read_text() == "new"
    assert out.startswith("Overwrote")
    assert out.display["previous_content"] == "old"
    assert out.display["file_existed"] is True


def test_write_file_refuses_a_folder_and_too_much(root):
    (root / "d").mkdir()
    assert _is_error(write_file_impl("d", "x", root))
    assert "limit" in write_file_impl("big.txt", "x" * 70_000, root)
    assert not (root / "big.txt").exists()


# ---- edit -----------------------------------------------------------------

def test_edit_replaces_once_and_reports_the_diff_occurrence(root):
    (root / "a.py").write_text("a = 1\nb = 2\nc = 3\n")
    out = edit_impl("a.py", "b = 2", "b = 20", root)
    assert (root / "a.py").read_text() == "a = 1\nb = 20\nc = 3\n"
    assert out.display["occurrences"] == [
        {"start_line": 2, "old_text": "b = 2", "new_text": "b = 20"}]


def test_edit_shows_whole_lines_around_a_partial_match(root):
    (root / "a.py").write_text("x = foo(1)\n")
    out = edit_impl("a.py", "foo", "bar", root)
    assert out.display["occurrences"][0] == {
        "start_line": 1, "old_text": "x = foo(1)", "new_text": "x = bar(1)"}


def test_edit_not_found_and_ambiguous_leave_the_file_alone(root):
    (root / "a.txt").write_text("x\nx\n")
    missing = edit_impl("a.txt", "zzz", "y", root)
    assert _is_error(missing) and "not found" in missing
    ambiguous = edit_impl("a.txt", "x", "y", root)
    assert _is_error(ambiguous) and "2 matches" in ambiguous
    assert "replace_all" in ambiguous
    assert (root / "a.txt").read_text() == "x\nx\n"


def test_edit_replace_all_changes_every_occurrence(root):
    (root / "a.txt").write_text("x\nmid\nx\n")
    out = edit_impl("a.txt", "x", "y", root, replace_all=True)
    assert (root / "a.txt").read_text() == "y\nmid\ny\n"
    assert [o["start_line"] for o in out.display["occurrences"]] == [1, 3]


def test_edit_input_checks(root):
    (root / "a.txt").write_text("x")
    assert "cannot be empty" in edit_impl("a.txt", "", "y", root)
    assert "identical" in edit_impl("a.txt", "x", "x", root)
    assert "does not exist" in edit_impl("nope.txt", "x", "y", root)


def test_edit_keeps_crlf_line_endings(root):
    (root / "w.txt").write_bytes(b"a\r\nb\r\n")
    edit_impl("w.txt", "a", "z", root)
    assert (root / "w.txt").read_bytes() == b"z\r\nb\r\n"


def test_edit_refuses_a_file_that_is_not_text(root):
    (root / "b.bin").write_bytes(b"\xff\xfe\x00\x80")
    assert "not valid UTF-8" in edit_impl("b.bin", "a", "b", root)


# ---- grep -----------------------------------------------------------------

def _tree(root):
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("import os\nneedle = 1\n")
    (root / "src" / "b.txt").write_text("a Needle here\n")
    for skipped in (".git", ".venv", "node_modules", "__pycache__"):
        (root / skipped).mkdir()
        (root / skipped / "x.py").write_text("needle = 'hidden'\n")
    (root / "bin.dat").write_bytes(b"\x00needle\x00")


def _grep_both_ways(monkeypatch, *args, **kwargs):
    """The same search with ripgrep (if installed) and without it."""
    results = {}
    if grep_tool.shutil.which("rg"):
        results["rg"] = grep_impl(*args, **kwargs)
    monkeypatch.setattr(grep_tool.shutil, "which", lambda name: None)
    results["python"] = grep_impl(*args, **kwargs)
    return results


def test_grep_finds_matches_and_skips_noise_with_or_without_rg(
        root, monkeypatch):
    _tree(root)
    for how, out in _grep_both_ways(monkeypatch, "needle", root).items():
        lines = sorted(str(out).split("\n\n")[0].splitlines())
        assert lines == ["src/a.py:2:needle = 1",
                         "src/b.txt:1:a Needle here"], how
        assert out.display["match_count"] == 2
        assert out.display["parsed_matches"][0]["line"] in (1, 2)


def test_grep_glob_and_smart_case(root, monkeypatch):
    _tree(root)
    for how, out in _grep_both_ways(monkeypatch, "Needle", root,
                                    glob="*.py").items():
        assert "No matches" in out, how         # capital N: case matters
    for how, out in _grep_both_ways(monkeypatch, "needle", root,
                                    glob="*.txt").items():
        assert str(out).strip() == "src/b.txt:1:a Needle here", how


def test_grep_caps_the_matches(root, monkeypatch):
    (root / "many.txt").write_text("hit\n" * 50)
    for how, out in _grep_both_ways(monkeypatch, "hit", root,
                                    max_matches=5).items():
        assert out.display["match_count"] == 5, how
        assert out.display["was_truncated"] is True, how
        assert "narrow the pattern" in out


def test_grep_bad_input_is_a_sentence(root):
    assert "empty" in grep_impl("  ", root)
    assert "not a valid regex" in grep_impl("(", root)
    assert "does not exist" in grep_impl("x", root, path="nope")


def test_grep_searches_one_file(root, monkeypatch):
    _tree(root)
    for how, out in _grep_both_ways(monkeypatch, "needle", root,
                                    path="src/a.py").items():
        assert str(out).strip() == "src/a.py:2:needle = 1", how


# ---- shell ----------------------------------------------------------------

def _py(code):
    # Quoted for both cmd.exe and sh: the code has no quotes of its own.
    return f'"{sys.executable}" -c "{code}"'


def test_shell_returns_stdout_stderr_and_exit_code(root):
    out = shell_impl(_py("import sys; print('out'); "
                         "print('err', file=sys.stderr)"), root)
    assert not _is_error(out)
    assert "out" in out and "err" in out
    assert out.display["stdout"].strip() == "out"
    assert out.display["stderr"].strip() == "err"
    assert out.display["exit_code"] == 0


def test_shell_runs_in_the_project_folder(root):
    out = shell_impl(_py("import os; print(os.getcwd())"), root)
    assert os.path.samefile(out.display["stdout"].strip(), root)


def test_shell_a_failing_command_reports_its_exit_code(root):
    out = shell_impl(_py("import sys; print('boom'); sys.exit(3)"), root)
    assert _is_error(out) and "code 3" in out and "boom" in out
    assert out.display["exit_code"] == 3 and out.display["failed"]


def test_shell_stdin_is_closed_so_a_prompt_does_not_hang(root):
    out = shell_impl(_py("import sys; print(repr(sys.stdin.read()))"), root,
                     timeout=30)
    assert "''" in out


def test_shell_caps_a_flood_of_output(root):
    out = shell_impl(_py("print('x' * 200000)"), root, timeout=30)
    assert out.display["truncated"] is True
    assert len(out.display["stdout"]) <= 30_000


def test_shell_timeout_kills_the_command_and_its_children(root):
    marker = root / "alive.txt"
    # The shell starts a python that sleeps then writes the marker. If only
    # the shell died, the grandchild would survive and write it.
    child = ("import time; time.sleep(3); "
             f"open(r'{marker}', 'w').write('x')")
    started = time.monotonic()
    # "&& echo" stops sh from exec-ing python directly: python must be a
    # CHILD of the shell, or killing the shell would kill it by accident.
    out = shell_impl(f'"{sys.executable}" -c "{child}" && echo done', root,
                     timeout=1)
    assert _is_error(out) and "timed out after 1 s" in out
    assert time.monotonic() - started < 2.9
    time.sleep(3.5)
    assert not marker.exists()


def test_shell_cancel_kills_the_command(root):
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    started = time.monotonic()
    out = shell_impl(_py("import time; time.sleep(30)"), root, timeout=60,
                     stop=stop)
    assert time.monotonic() - started < 10
    assert out == "Error: cancelled by the caller."


def test_shell_already_cancelled_does_not_start(root):
    stop = threading.Event()
    stop.set()
    assert shell_impl("echo hi", root, stop=stop).startswith("Error:")


def test_shell_timeout_is_clamped(root):
    # A huge timeout is accepted and clamped, a bogus one falls back.
    assert not _is_error(shell_impl(_py("print(1)"), root, timeout=99999))
    assert not _is_error(shell_impl(_py("print(1)"), root, timeout="soon"))
    assert "empty" in shell_impl("  ", root)


# ---- wiring: schemas, arg models, permissions, the factory ---------------

def _cfg(tools, **extra):
    return WorkflowConfig(model={"name": "m"}, tools=tools, **extra)


def test_the_five_tools_validate_as_builtins_and_build(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tools = factory.build_tools(_cfg([{"name": n} for n in NAMES]))
    assert list(tools) == NAMES
    assert all(t["kind"] == "builtin" for t in tools.values())
    assert set(NAMES) == set(CODING_TOOLS)


@pytest.mark.parametrize("name", NAMES)
def test_every_engine_can_derive_arguments_from_the_schema(name):
    schema = factory.BUILTIN_TOOL_SCHEMAS[name]
    assert schema["function"]["name"] == name
    model = args_model_from_schema(name, schema)
    required = schema["function"]["parameters"]["required"]
    assert all(model.model_fields[r].is_required() for r in required)
    raw = json_schema_for_tool(schema)
    assert raw["type"] == "object" and set(required) <= set(raw["properties"])


def test_args_models_accept_the_model_calling_with_only_required_fields():
    for name in NAMES:
        schema = factory.BUILTIN_TOOL_SCHEMAS[name]
        model = args_model_from_schema(name, schema)
        needed = {r: "x" for r in schema["function"]["parameters"]["required"]}
        assert model(**needed)


def test_default_permissions_ask_for_the_three_that_change_things():
    assert default_permission("builtin", None, "read_file") == "allow"
    assert default_permission("builtin", None, "grep") == "allow"
    for name in ("write_file", "edit", "shell"):
        assert default_permission("builtin", None, name) == "ask"
    # Callers that do not pass a name, and other kinds, are as before.
    assert default_permission("builtin") == "allow"
    assert default_permission("mcp_stdio", None, "shell") == "allow"


def test_the_factory_applies_the_defaults_and_a_workflow_can_override(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tools = factory.build_tools(_cfg([
        {"name": "read_file"}, {"name": "edit"},
        {"name": "shell", "permission": "allow"},
        {"name": "write_file", "permission": "deny"}]))
    assert {n: t["permission"] for n, t in tools.items()} == {
        "read_file": "allow", "edit": "ask", "shell": "allow",
        "write_file": "deny"}


def test_headless_ask_is_a_refusal_and_nothing_is_written(tmp_path,
                                                          monkeypatch):
    monkeypatch.chdir(tmp_path)
    bus = EventBus()
    tools = factory.observe_tools(
        factory.build_tools(_cfg([{"name": "write_file"}, {"name": "shell"}]),
                            events=bus), bus)
    refused = tools["write_file"]["function"](file_path="a.txt", content="x")
    assert "needs a person's approval" in refused
    assert not (tmp_path / "a.txt").exists()
    assert "needs a person's approval" in tools["shell"]["function"](
        command="echo hi")
    bus.approver = lambda name, args, reason: True
    assert tools["write_file"]["function"](file_path="a.txt", content="x")
    assert (tmp_path / "a.txt").read_text() == "x"


def test_the_project_folder_is_the_working_directory_at_build_time(
        tmp_path, monkeypatch):
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "f.txt").write_text("in")
    (tmp_path / "f.txt").write_text("out")
    monkeypatch.chdir(tmp_path / "proj")
    tools = factory.build_tools(_cfg([{"name": "read_file"}]))
    monkeypatch.chdir(tmp_path)           # moving later changes nothing
    assert "1\u2192in" in tools["read_file"]["function"](file_path="f.txt")


def test_cancelling_the_bus_kills_a_running_shell_command(tmp_path,
                                                          monkeypatch):
    monkeypatch.chdir(tmp_path)
    bus = EventBus()
    tools = factory.build_tools(
        _cfg([{"name": "shell", "permission": "allow"}]), events=bus)
    threading.Timer(0.5, bus.stop.set).start()
    started = time.monotonic()
    out = tools["shell"]["function"](
        command=_py("import time; time.sleep(30)"), timeout=60)
    assert time.monotonic() - started < 10
    assert out.startswith("Error:")


def test_observed_reports_the_display_and_marks_failures():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda e: seen.append(e) if e.type == TOOL_ENDED else None)
    good = observed("t", lambda: Shown("fine", {"kind": "x", "n": 1}), bus)
    bad = observed("t", lambda: Shown("Error: no", {"failed": True}), bus)
    plain = observed("t", lambda: "plain", bus)
    assert good() == "fine" and isinstance(good(), str)
    bad()
    plain()
    ok, _, failed, none = seen
    assert ok.data["display"] == {"kind": "x", "n": 1}
    assert failed.data["status"] == "error"
    assert "display" not in none.data and none.data["status"] == "ok"


def test_a_native_agent_edits_a_file_through_the_loop(tmp_path, monkeypatch):
    """The loop end to end: model asks for edit, the file changes."""
    from tests.conftest import FakeLLMProvider, make_tool_call, reply
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("hello world")
    bus = EventBus()
    bus.approver = lambda name, args, reason: True
    tools = factory.observe_tools(
        factory.build_tools(_cfg([{"name": "edit"}]), events=bus), bus)
    from ovat.agent.loop import AgentLoop
    llm = FakeLLMProvider([
        reply("tool_calls", tool_calls=[make_tool_call("1", "edit", {
            "file_path": "a.txt", "old_string": "world",
            "new_string": "there"})]),
        reply("stop", content="done")])
    agent = AgentLoop(llm=llm, tools=tools, engine_name="native",
                      stop=bus.stop, events=bus)
    assert agent.run("fix it") == "done"
    assert (tmp_path / "a.txt").read_text() == "hello there"
    sent = str(llm.calls[1]["messages"])
    assert "replaced 1 occurrence" in sent
