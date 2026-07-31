# tests/test_plano_example.py
"""The plano gateway example: OVAT talking to OVMS THROUGH plano.

No plano binary is needed here. What is asserted is that the two config files
agree with each other and with the answer to the mentor's spike question,
because a demo that fails on a mismatched port is a demo that fails.
"""
import pytest

yaml = pytest.importorskip("yaml")

from ovat.config.workflow import load_workflow

CONFIG = "examples/plano/plano-config.yaml"
WORKFLOW = "examples/plano/workflow.yml"


def _plano() -> dict:
    with open(CONFIG, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_the_workflow_points_at_plano_not_at_ovms():
    """The whole point of the example: OVAT's ovms_url is just a URL, and
    plano's job is to look exactly like the one it replaces. If the workflow
    still pointed at 8000 the gateway would be bypassed entirely."""
    config = load_workflow(WORKFLOW)
    listener = _plano()["listeners"][0]
    assert str(listener["port"]) in config.model.ovms_url, (
        f"workflow points at {config.model.ovms_url} but plano listens on "
        f"{listener['port']}")


def test_the_v3_prefix_is_set_which_is_the_whole_spike_answer():
    """Mentor Ravi's question: plano assumes upstreams serve /v1, OVMS serves
    /v3. Upstream's own test (test_target_endpoint_with_base_url_prefix in
    crates/hermesllm/src/clients/endpoints.rs) proves base_url_path_prefix
    REPLACES the default. One config line, no fork."""
    provider = _plano()["model_providers"][0]
    assert provider.get("base_url_path_prefix") == "/v3"


def test_the_provider_points_at_the_port_ovat_serve_uses():
    provider = _plano()["model_providers"][0]
    assert provider["port"] == 8000


def test_the_model_name_matches_on_both_sides():
    """plano routes by model name; a mismatch is a 404 at demo time."""
    provider = _plano()["model_providers"][0]
    assert provider["model"] == load_workflow(WORKFLOW).model.name


def test_tracing_is_on_so_the_gateway_earns_its_hop():
    """The reason to accept an extra network hop: every request becomes an
    OpenTelemetry span, which is the W9-W10 telemetry item by configuration
    rather than by writing an exporter."""
    tracing = _plano().get("tracing")
    assert tracing, "no tracing block; the extra hop buys nothing"
    assert tracing["random_sampling"] == 100
