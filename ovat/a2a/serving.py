# ovat/a2a/serving.py
"""Everything `ovat a2a serve` decides before it opens a port. No SDK.

Kept out of the CLI so it can be tested without a socket, and out of
sdk/server.py so none of these rules depend on the SDK:
- the safety rules for where the server may listen;
- building the first agent (which also proves the workflow builds);
- which tools are exposed, and the card that describes them;
- who may call, and the log of what left.
"""
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from ovat.a2a import exposure as exposure_mod
from ovat.a2a import tokens as tokens_mod
from ovat.a2a.audit import AuditLog
from ovat.a2a.card import card_dict
from ovat.a2a.service import A2AService
from ovat.agent.factory import build_agent, close_agent
from ovat.config.workflow import A2AServerConfig, WorkflowConfig

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


def check_listen_address(host: str, public_url: str | None, tls: bool,
                         allow_plain_http: bool) -> None:
    """Raise ValueError, with the fix, if this address is unsafe.

    Off this machine: callers need a URL to put on the card (public_url),
    and the token must not cross the network in clear text unless the owner
    said so out loud. The spec requires HTTPS in production (section 7.1).
    """
    if host in LOOPBACK:
        return
    if not public_url:
        raise ValueError(
            f"listening on {host} reaches other machines, so set "
            f"a2a.server.public_url (or --public-url) to the address they "
            f"will use. It goes on the card, and only that host name is "
            f"accepted in requests.")
    if not tls and not allow_plain_http:
        raise ValueError(
            f"listening on {host} without TLS would send the token over the "
            f"network in clear text. Pass --tls-cert and --tls-key, or "
            f"--allow-plain-http if this network is trusted. For two "
            f"machines, an SSH tunnel to 127.0.0.1 is safer than either.")


def default_name(workflow_path: str) -> str:
    """examples/rag/workflow.yml -> "rag"; team-docs.yml -> "team-docs"."""
    p = Path(workflow_path)
    if p.stem == "workflow" and p.parent.name:
        return p.parent.name
    return p.stem


@dataclass
class Prepared:
    service: A2AService
    card: dict
    callers: dict[str, str]                 # token -> caller name
    exposure: exposure_mod.Exposure
    url: str
    allowed_hosts: list[str] = field(default_factory=list)
    audit_path: str | None = None


def prepare(config: WorkflowConfig, settings: A2AServerConfig,
            workflow_path: str, version: str, tls: bool = False,
            skip_rag: bool = False, build=build_agent,
            token_file: Path = tokens_mod.DEFAULT_TOKEN_FILE,
            dry_run: bool = False) -> Prepared:
    """Build what the server needs. Raises ValueError/RuntimeError with a
    sentence on any problem, before any port is opened.

    dry_run (`ovat a2a card`): decide and describe, but create no token
    file and no audit log, since nothing will be served."""
    scheme = "https" if tls else "http"
    url = settings.public_url or f"{scheme}://{settings.host}:{settings.port}/"

    # The first agent is built NOW: it proves the workflow builds (models,
    # index, MCP commands), and it is what the exposure is decided from.
    decided: dict = {}

    def select(tools: dict) -> dict:
        if "exposure" not in decided:
            decided["exposure"] = exposure_mod.decide(
                tools, settings.expose_tools)
            decided["tools"] = tools
        return decided["exposure"].select(tools)

    first = build(config, skip_rag=skip_rag, select_tools=select)
    exp = decided["exposure"]
    try:
        callers = ({} if dry_run
                   else tokens_mod.callers(settings.tokens, token_file))
        audit = (AuditLog(settings.audit_log)
                 if settings.audit_log and not dry_run else None)
    except Exception:
        close_agent(first)
        raise

    spare = [first]

    def build_one():
        # The first conversation reuses the agent built above, so the
        # first caller does not wait for a second model load.
        if spare:
            return spare.pop()
        return build(config, skip_rag=skip_rag, select_tools=exp.select)

    service = A2AService(build_one, settings, audit=audit)
    card = card_dict(config, settings, exp.select(decided["tools"]), url,
                     settings.name or default_name(workflow_path), version)
    hosts = ["127.0.0.1", "localhost"]
    if settings.host not in LOOPBACK and settings.host != "0.0.0.0":
        hosts.append(settings.host)
    if settings.public_url:
        public_host = urlparse(settings.public_url).hostname
        if public_host:
            hosts.append(public_host)
    return Prepared(service=service, card=card, callers=callers,
                    exposure=exp, url=url,
                    allowed_hosts=list(dict.fromkeys(hosts)),
                    audit_path=settings.audit_log)
