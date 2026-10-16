# ovat/cli/a2a_cmd.py
"""`ovat a2a ...`: serve this workflow to other agents, or ask one.

Imports nothing from the SDK at module level: main.py registers this
sub-app on every run, and `ovat --help` must work without the [a2a] extra.
Each command checks for the SDK first and says how to install it.
"""
from typing import Optional

import typer

from ovat.cli.ui import console, esc

rprint = console.print

a2a_app = typer.Typer(
    help="Agent2Agent: offer this workflow to other agents, or ask one.",
    no_args_is_help=True,
)


def _need_sdk() -> None:
    from ovat.a2a.sdk import require_sdk
    try:
        require_sdk()
    except RuntimeError as exc:
        rprint(f"[red]{esc(str(exc))}[/red]")
        raise typer.Exit(code=1)


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("ovat")
    except PackageNotFoundError:
        return "0"


def _settings(cfg, host, port, public_url):
    """The workflow's a2a.server section, with any flags laid over it."""
    from ovat.config.workflow import A2AConfig
    server = (cfg.a2a or A2AConfig()).server
    update = {k: v for k, v in (("host", host), ("port", port),
                                ("public_url", public_url)) if v is not None}
    return server.model_copy(update=update) if update else server


def _prepare(config, make_settings, tls=False, dry_run=False):
    from ovat.a2a.serving import prepare
    from ovat.cli.main import _load_config
    cfg = _load_config(config)
    settings = make_settings(cfg)
    try:
        return cfg, settings, prepare(cfg, settings, config, _version(),
                                      tls=tls, skip_rag=dry_run,
                                      dry_run=dry_run)
    except (ValueError, RuntimeError, OSError) as exc:
        rprint(f"[red]Cannot serve {esc(config)}:[/red] {esc(str(exc))}")
        raise typer.Exit(code=1)


def _print_exposure(prepared) -> None:
    exp = prepared.exposure
    rprint(f"Exposed: {esc(', '.join(exp.exposed) or 'no tools (chat only)')}")
    for name, why in exp.hidden.items():
        rprint(f"[dim]Not exposed: {esc(name)} ({esc(why)})[/dim]")
    for warning in exp.warnings:
        rprint(f"[yellow]Warning:[/yellow] {esc(warning)}")


@a2a_app.command()
def serve(
    config: str = typer.Argument(..., help="The workflow to offer."),
    host: Optional[str] = typer.Option(None, help="Address to listen on."),
    port: Optional[int] = typer.Option(None, help="Port to listen on."),
    public_url: Optional[str] = typer.Option(
        None, help="URL other agents use, if not host:port."),
    tls_cert: Optional[str] = typer.Option(None, help="TLS certificate file."),
    tls_key: Optional[str] = typer.Option(None, help="TLS private key file."),
    allow_plain_http: bool = typer.Option(
        False, help="Allow plain HTTP off this machine (token in clear text)."),
):
    """Offer a workflow to other agents over A2A. Ctrl-C stops it."""
    _need_sdk()
    from ovat.a2a.serving import check_listen_address
    if bool(tls_cert) != bool(tls_key):
        rprint("[red]--tls-cert and --tls-key go together.[/red]")
        raise typer.Exit(code=1)
    tls = bool(tls_cert)

    def settings(cfg):
        s = _settings(cfg, host, port, public_url)
        try:
            check_listen_address(s.host, s.public_url, tls, allow_plain_http)
        except ValueError as exc:
            rprint(f"[red]Not starting:[/red] {esc(str(exc))}")
            raise typer.Exit(code=1)
        return s

    cfg, s, prepared = _prepare(config, settings, tls=tls)
    import uvicorn
    from ovat.a2a.sdk.server import build_app

    where = "this machine only" if s.host in ("127.0.0.1", "localhost",
                                              "::1") else "the network"
    rprint(f'A2A agent "{esc(prepared.card["name"])}" is listening on '
           f"{esc(prepared.url)} ({where})")
    rprint(f"Card:   {esc(prepared.url.rstrip('/'))}"
           f"/.well-known/agent-card.json")
    if not s.tokens:
        rprint("Token:  .ovat/a2a/token (callers send it as "
               "\"Authorization: Bearer ...\"; 'ovat a2a token' prints it)")
    else:
        rprint(f"Callers: {esc(', '.join(s.tokens))}")
    rprint(f"Engine: {esc(cfg.agent.type)} | model {esc(cfg.model.name)} "
           f"on {esc(cfg.model.device)}")
    _print_exposure(prepared)
    rprint(f"Audit:  {esc(prepared.audit_path or 'off')}")
    rprint("[dim]Ctrl-C to stop.[/dim]")
    app = build_app(prepared.service, prepared.card, prepared.callers,
                    prepared.allowed_hosts)
    uvicorn.run(app, host=s.host, port=s.port, log_level="warning",
                ssl_certfile=tls_cert, ssl_keyfile=tls_key)


@a2a_app.command()
def card(
    config: str = typer.Argument(..., help="The workflow to describe."),
):
    """Print the Agent Card a workflow would serve. Starts nothing."""
    _need_sdk()
    from ovat.a2a.sdk.server import card_json
    _, _, prepared = _prepare(
        config, lambda cfg: _settings(cfg, None, None, None), dry_run=True)
    try:
        console.print_json(card_json(prepared.card))
        _print_exposure(prepared)
    finally:
        prepared.service.close()


@a2a_app.command()
def call(
    url: str = typer.Argument(..., help="The other agent's base URL."),
    message: str = typer.Argument(..., help="What to ask it."),
    token_env: Optional[str] = typer.Option(
        None, help="Environment variable holding its bearer token."),
    timeout: float = typer.Option(900.0, help="Seconds to wait."),
):
    """Ask another agent one question and print its answer."""
    _need_sdk()
    from ovat.a2a.protocol import PeerError, TaskState
    from ovat.a2a.sdk.client import SDKPeer
    from ovat.a2a.tokens import peer_token
    token = peer_token(url, token_env)
    if token_env and not token:
        rprint(f"[red]{esc(token_env)} is not set.[/red]")
        raise typer.Exit(code=1)
    peer = SDKPeer(url, token=token or None, timeout=timeout)
    try:
        remote = peer.card()
        reply = peer.send(message, on_status=lambda text: rprint(
            f"[dim]{esc(remote.name)}  {esc(text)}[/dim]"))
    except PeerError as exc:
        rprint(f"[red]Could not ask {esc(url)}:[/red] {esc(exc.kind)}"
               + (f" ({esc(exc.detail)})" if exc.detail else ""))
        raise typer.Exit(code=1)
    finally:
        peer.close()
    if reply.state is not TaskState.COMPLETED:
        rprint(f"[red]{esc(reply.state.value)}:[/red] {esc(reply.text)}")
        raise typer.Exit(code=1)
    rprint(esc(reply.text))
    from ovat.tools.a2a_call import remote_totals
    totals = remote_totals(reply.metadata)
    if totals:
        rprint(f"[dim]{totals.get('turns', '?')} turns, "
               f"{totals.get('tool_calls', '?')} tool calls, "
               f"{totals.get('latency_s', '?')} s[/dim]")


@a2a_app.command()
def token(
    rotate: bool = typer.Option(False, help="Replace it; old callers stop "
                                            "working."),
):
    """Print the generated token callers send (made on first use)."""
    from ovat.a2a import tokens
    value = tokens.rotate() if rotate else tokens.load_or_create()
    rprint(esc(value))
