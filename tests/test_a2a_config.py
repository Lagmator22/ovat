# tests/test_a2a_config.py
"""The A2A parts of workflow.yml: strict, optional, and safe by default.

None of these need the [a2a] extra: the schema is plain pydantic, so a
workflow can be validated (and `ovat doctor` can read it) on a machine that
never installed the SDK.
"""
import glob

import pytest
from pydantic import ValidationError

from ovat.config.workflow import WorkflowConfig, load_workflow


def _cfg(**extra):
    return WorkflowConfig(model={"name": "m"}, **extra)


def test_a_workflow_without_a2a_is_unchanged():
    assert _cfg().a2a is None


def test_every_shipped_example_still_validates():
    paths = glob.glob("examples/**/*.yml", recursive=True)
    assert paths
    for path in paths:
        load_workflow(path)


def test_server_defaults_are_the_safe_ones():
    server = _cfg(a2a={"server": {}}).a2a.server
    assert server.host == "127.0.0.1"     # this machine only
    assert server.expose_sources == "names"   # no full paths leave
    assert server.audit_log                    # every answer is logged
    assert server.max_hops >= 1


def test_an_unknown_server_key_is_an_error():
    with pytest.raises(ValidationError, match="prot"):
        _cfg(a2a={"server": {"prot": 9100}})


def test_a_skill_needs_the_fields_the_spec_requires():
    with pytest.raises(ValidationError, match="description"):
        _cfg(a2a={"server": {"skills": [{"id": "s", "name": "S"}]}})
    skill = _cfg(a2a={"server": {"skills": [
        {"id": "s", "name": "S", "description": "d"}]}}).a2a.server.skills[0]
    assert skill.tags                     # never empty: the spec requires it


def test_a_port_out_of_range_is_an_error():
    with pytest.raises(ValidationError):
        _cfg(a2a={"server": {"port": 70000}})


def test_an_a2a_tool_needs_an_http_url():
    with pytest.raises(ValidationError, match="url"):
        _cfg(tools=[{"name": "peer", "type": "a2a"}])
    with pytest.raises(ValidationError, match="url"):
        _cfg(tools=[{"name": "peer", "type": "a2a", "url": "ftp://x"}])
    tool = _cfg(tools=[{"name": "peer", "type": "a2a",
                        "url": "http://127.0.0.1:9100",
                        "token_env": "PEER_TOKEN"}]).tools[0]
    assert tool.url == "http://127.0.0.1:9100"


def test_a2a_fields_on_another_tool_type_are_named_errors():
    with pytest.raises(ValidationError, match="url"):
        _cfg(tools=[{"name": "search_docs", "url": "http://x"}])
    with pytest.raises(ValidationError, match="token_env"):
        _cfg(tools=[{"name": "search_docs", "token_env": "X"}])


def test_a_token_never_goes_in_the_yaml_itself():
    # tokens: maps a caller name to the NAME of an environment variable.
    # A value that looks like a secret rather than a variable name is
    # refused, so a pasted token cannot end up committed in a workflow.
    with pytest.raises(ValidationError, match="environment variable"):
        _cfg(a2a={"server": {"tokens": {"planner": "s3cr3t-t0ken/abc+="}}})
    ok = _cfg(a2a={"server": {"tokens": {"planner": "OVAT_A2A_PLANNER"}}})
    assert ok.a2a.server.tokens == {"planner": "OVAT_A2A_PLANNER"}
