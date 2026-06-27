# ovat/cli/main.py
"""Layer 1 (entry point): the ovat command line, one YAML + one command.

This is the face of the whole toolkit. pyproject.toml points the `ovat` command
here (ovat = "ovat.cli.main:app"). typer turns each function below into a
subcommand automatically, using the type hints to parse arguments, so
`def run(config: str, ...)` becomes `ovat run <config> ...` for free.

The headline command is `run`: load a workflow YAML, build the agent, ask it
my question, print the answer. That single line is the midterm demo.
"""
import typer
from rich import print as rprint

from ovat.agent.factory import build_agent
from ovat.config.workflow import load_workflow

app = typer.Typer(
    help="OVAT: run an OpenVINO agent from one YAML + one command.",
    add_completion=False,
)

# A starter workflow I write out for `ovat init`, so a new user has something
# that already works to edit instead of a blank file.
_STARTER_YAML = """\
# OVAT workflow. Edit this, then run:  ovat run workflow.yml --input "..."
model:
  name: Qwen3-8B-int4-ov
  device: GPU
  ovms_url: http://localhost:8000/v3
  tool_parser: hermes3
  # Only used by `ovat serve` to start OVMS and locate the model:
  source_model: OpenVINO/Qwen3-8B-int4-ov
  model_repository_path: models     # set to an absolute path if needed, e.g. C:\\Users\\you\\models

tools:
  - name: search_docs
    type: builtin
  - name: transcribe
    type: builtin

agent:
  type: native
  max_iterations: 10
  system_prompt: "You are a helpful assistant that uses tools when needed."
"""


@app.command()
def run(
    config: str = typer.Argument(..., help="Path to a workflow YAML."),
    input: str = typer.Option(..., "--input", "-i", help="Your question for the agent."),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="Build the agent and show it, but do not call the model."),
):
    """Run the agent described by CONFIG against your input."""
    # Step 1: YAML -> validated config. A bad file fails loudly right here.
    cfg = load_workflow(config)
    # Step 2: config -> a fully wired agent (LLM + tools + loop).
    agent = build_agent(cfg)

    # dry-run lets me prove the wiring on any machine, even with no OVMS server.
    if dry_run:
        rprint(f"[green]Built agent[/green]  model={cfg.model.name}  "
               f"tools={list(agent.tools)}  max_iterations={agent.max_iterations}")
        rprint("[yellow]dry-run:[/yellow] not calling the model.")
        raise typer.Exit()

    # Step 3: actually run. This needs a live OVMS server to answer.
    try:
        answer = agent.run(input)
    except Exception as exc:
        rprint(f"[red]Error talking to OVMS at {cfg.model.ovms_url}[/red]: {exc}")
        raise typer.Exit(code=1)
    rprint(answer)


@app.command()
def init(
    path: str = typer.Argument("workflow.yml", help="Where to write the starter YAML."),
):
    """Write a starter workflow.yml you can edit."""
    import os
    if os.path.exists(path):
        rprint(f"[red]Refusing to overwrite existing file:[/red] {path}")
        raise typer.Exit(code=1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_STARTER_YAML)
    rprint(f"[green]Wrote starter workflow to[/green] {path}")


@app.command()
def models(
    action: str = typer.Argument("list", help="list or pull"),
    source_model: str = typer.Option(None, "--source-model",
                                     help="Hugging Face id to pull, e.g. OpenVINO/Qwen3-8B-int4-ov."),
):
    """List or pull OVMS models (needs the ovms binary, so runs on the AI PC)."""
    from ovat.core.model_manager import ModelManager
    mgr = ModelManager()
    if action == "list":
        for name in mgr.list_models():
            rprint(name)
    elif action == "pull":
        if not source_model:
            rprint("[red]pull needs --source-model[/red]")
            raise typer.Exit(code=1)
        rprint(mgr.pull(source_model))
    else:
        rprint(f"[red]Unknown action '{action}'. Use list or pull.[/red]")
        raise typer.Exit(code=1)


@app.command()
def serve(
    config: str = typer.Argument(..., help="Workflow YAML whose model OVMS should serve."),
):
    """Start OVMS serving the model from a workflow YAML (runs on the AI PC)."""
    from ovat.core.model_server import ModelServer
    cfg = load_workflow(config)
    server = ModelServer(
        model_name=cfg.model.name,
        source_model=cfg.model.source_model,
        model_repository_path=cfg.model.model_repository_path,
        device=cfg.model.device,
        tool_parser=cfg.model.tool_parser,
        reasoning_parser=cfg.model.reasoning_parser,
    )
    rprint(f"[green]Starting OVMS[/green] for {cfg.model.name} on {cfg.model.device} ...")
    server.start()
    if server.wait_until_ready():
        rprint(f"[green]OVMS is ready[/green] at {server.base_url}")
    else:
        rprint("[red]OVMS did not become ready in time.[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
