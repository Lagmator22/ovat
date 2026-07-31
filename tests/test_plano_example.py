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


def test_the_v3_path_is_in_the_base_url_which_is_the_spike_answer():
    """Mentor Ravi's question: plano defaults to calling upstreams at /v1,
    OVMS serves its OpenAI API under /v3.

    There is NO separate prefix field to set. plano's CLI parses base_url and
    lifts the path into base_url_path_prefix itself (config_generator.py,
    get_endpoint_and_port), and its schema REJECTS base_url_path_prefix as an
    unexpected property. An earlier version of this file set it by hand and
    `planoai up` failed validation outright.

    So the whole fix is /v3 in this URL, and it is the same value as OVAT's
    own model.ovms_url.
    """
    base_url = _plano()["model_providers"][0]["base_url"]
    assert base_url.rstrip("/").endswith("/v3"), base_url
    assert ":8000" in base_url, "must point at the port `ovat serve` uses"


def test_the_config_validates_against_planos_own_schema():
    """The test that would have caught the failure on the AI PC. plano ships
    its schema inside the wheel, so this checks the real contract rather than
    a guess at it."""
    import importlib.util
    from pathlib import Path

    # Ask the installed package where it is, rather than searching the disk:
    # a filesystem walk is slow and trips over unreadable paths.
    spec = importlib.util.find_spec("planoai")
    schema_path = None
    if spec and spec.submodule_search_locations:
        candidate = (Path(list(spec.submodule_search_locations)[0])
                     / "data" / "plano_config_schema.yaml")
        if candidate.is_file():
            schema_path = candidate
    if schema_path is None:
        # planoai installs as an isolated uv tool, so it is usually not
        # importable from OVAT's venv. Look where uv puts it.
        for root in (Path.home() / ".local/share/uv/tools/planoai",):
            found = list(root.rglob("plano_config_schema.yaml"))
            if found:
                schema_path = found[0]
                break
    if schema_path is None:
        pytest.skip("planoai is not installed here")
    with open(schema_path, encoding="utf-8") as handle:
        schema = yaml.safe_load(handle)

    config = _plano()
    # model_providers is where the failure happened and where the schema is a
    # plain array of objects. listeners is a oneOf union of several shapes,
    # so checking it this way would need the union resolved and is not worth
    # the complexity for a config with one listener in it.
    allowed = set(schema["properties"]["model_providers"]["items"]
                  ["properties"])
    for entry in config["model_providers"]:
        unexpected = set(entry) - allowed
        assert not unexpected, (
            f"plano will reject these keys: {sorted(unexpected)}. Its schema "
            f"accepts only {sorted(allowed)}")
    tracing_allowed = set(schema["properties"]["tracing"]["properties"])
    assert not set(config.get("tracing", {})) - tracing_allowed


def test_the_model_name_carries_a_provider_prefix_plano_can_split():
    """plano REQUIRES <provider>/<model_id> and raises "Invalid model name"
    without the slash (config_generator.py: model_name.split("/"), then
    `if len(tokens) < 2: raise`). An earlier version of this file had a bare
    model name and `planoai up` refused to start."""
    model = _plano()["model_providers"][0]["model"]
    assert "/" in model, f"plano will reject a bare model name: {model!r}"


def test_the_part_after_the_prefix_is_what_ovms_actually_serves():
    """plano strips the prefix (model_id = "/".join(tokens[1:])) and sends
    only the remainder upstream, so the half after the slash has to be the
    name OVMS serves or every request is a 404 at demo time."""
    served = _plano()["model_providers"][0]["model"].split("/", 1)[1]
    assert served == load_workflow(WORKFLOW).model.name


def test_an_unsupported_provider_supplies_what_plano_demands_of_it():
    """"ovms" is not one of plano's built-in providers. For an unsupported
    one it requires BOTH base_url and provider_interface, and that pair is
    the supported way to point plano at any OpenAI-compatible server it has
    never heard of."""
    provider = _plano()["model_providers"][0]
    assert provider.get("base_url")
    assert provider.get("provider_interface") == "openai"


def test_tracing_is_on_so_the_gateway_earns_its_hop():
    """The reason to accept an extra network hop: every request becomes an
    OpenTelemetry span, which is the W9-W10 telemetry item by configuration
    rather than by writing an exporter."""
    tracing = _plano().get("tracing")
    assert tracing, "no tracing block; the extra hop buys nothing"
    assert tracing["random_sampling"] == 100
    # `planoai obs` listens on 4317 and says so at startup; a mismatch here
    # means the dashboard sits at "waiting for spans" forever.
    assert "4317" in tracing["opentracing_grpc_endpoint"]


def test_the_workflow_uses_planos_front_door_path_not_ovms_own():
    """This looks wrong and is correct. There are two different paths:

        ovat  -> POST /v1/chat/completions -> plano :12000
        plano -> POST /v3/chat/completions -> OVMS  :8000

    Clients always call plano at /v1; that is its own front door and is not
    configurable. The UPSTREAM path is what base_url sets. If the workflow
    pointed at :8000/v3 the gateway would be bypassed and the example would
    prove nothing at all.
    """
    url = load_workflow(WORKFLOW).model.ovms_url
    assert url.rstrip("/").endswith("/v1"), url
    upstream = _plano()["model_providers"][0]["base_url"]
    assert upstream.rstrip("/").endswith("/v3"), upstream
    assert url != upstream, "the two hops must not be the same address"


def test_the_readme_states_the_windows_limitation():
    """plano has no Windows build, so `planoai up` fails before reading the
    config. Someone on the AI PC must find that in the README rather than by
    debugging a config that was never the problem."""
    with open("examples/plano/README.md", encoding="utf-8") as handle:
        readme = handle.read()
    assert "Unsupported platform windows" in readme
    assert "--docker" in readme
