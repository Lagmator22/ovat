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

# RAG for the search_docs tool. Run `ovat index <folder> workflow.yml` first to
# fill the index, then ask questions with `ovat run`. Swap a provider string to
# change a backend; no code changes, only this YAML.
rag:
  embeddings:
    provider: genai                 # genai (local) or ovms (server /v3)
    model: models/bge-small-en-v1.5 # OpenVINO embedding model folder on disk
    device: CPU                     # CPU or NPU on the AI PC
    dim: 384
  retriever:
    provider: sqlite-vec
    db_path: ovat_index.db
  chunk:
    size: 512
    overlap: 64
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
    # Step 2: config -> a fully wired agent (LLM + tools + loop). dry-run skips
    # loading the RAG model so the preview works on any machine.
    agent = build_agent(cfg, skip_rag=dry_run)

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
def index(
    folder: str = typer.Argument(..., help="Folder of .txt/.md documents to index."),
    config: str = typer.Argument(..., help="Workflow YAML whose rag: section to use."),
):
    """Index a folder of documents so search_docs can find them.

    This reads the rag: section of your workflow, builds the embedder and the
    vector store it names, chunks every text file under FOLDER, and stores the
    chunks. After this, `ovat run` can answer questions from those documents.
    """
    from ovat.agent.factory import build_rag
    from ovat.rag.indexer import index_folder

    cfg = load_workflow(config)
    if cfg.rag is None:
        rprint("[red]This workflow has no [bold]rag:[/bold] section.[/red] "
               "Add one (embeddings + retriever) before indexing.")
        raise typer.Exit(code=1)

    # Building the retriever loads the embedding model. If that model is not on
    # disk yet, say so plainly instead of dumping a pipeline traceback.
    try:
        retriever = build_rag(cfg)
    except Exception as exc:
        rprint(f"[red]Could not build the embedder/retriever:[/red] {exc}")
        rprint("[yellow]Tip:[/yellow] make sure the embeddings model in "
               f"[bold]{cfg.rag.embeddings.model}[/bold] exists on disk.")
        raise typer.Exit(code=1)

    rprint(f"[green]Indexing[/green] {folder} -> {cfg.rag.retriever.db_path} ...")
    try:
        summary = index_folder(
            folder, retriever,
            size=cfg.rag.chunk.size, overlap=cfg.rag.chunk.overlap,
        )
    except FileNotFoundError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    rprint(f"[green]Indexed[/green] {summary['chunks']} chunks "
           f"from {summary['files']} files.")


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
    try:
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
    except FileNotFoundError:
        # The ovms binary is not on PATH. Tell the user plainly instead of
        # dumping a raw subprocess traceback.
        rprint("[red]Could not find the 'ovms' binary on PATH.[/red] "
               "Install OVMS or add its folder to PATH (Windows: run setupvars first).")
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
    try:
        server.start()
    except FileNotFoundError:
        # ovms binary not on PATH. Clean message instead of a raw traceback.
        rprint("[red]Could not find the 'ovms' binary on PATH.[/red] On Windows, run "
               "setupvars.bat and add the OVMS folder to PATH before 'ovat serve'.")
        raise typer.Exit(code=1)
    if server.wait_until_ready():
        rprint(f"[green]OVMS is ready[/green] at {server.base_url}")
    else:
        rprint("[red]OVMS did not become ready in time.[/red]")
        raise typer.Exit(code=1)


@app.command()
def doctor(
    config: str = typer.Argument(None, help="Optional workflow YAML to validate too."),
):
    """Check the setup: Python, dependencies, devices, OVMS, and a config.

    Every row is a real check. Green means good, yellow is a heads-up that does
    not block anything, red is something to fix. Pass a workflow to also validate
    it and see whether its model and OVMS look ready.
    """
    from rich.table import Table

    from ovat.cli import diagnostics
    from ovat.cli.ui import banner, console, status_text

    banner("environment & workflow diagnostics")
    checks = diagnostics.run_checks(config)

    table = Table(header_style="ovat.header", border_style="ovat.dim",
                  expand=False)
    table.add_column("Check", style="ovat.cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail", style="ovat.dim")
    failures = 0
    for c in checks:
        if c.status == diagnostics.FAIL:
            failures += 1
        table.add_row(c.name, status_text(c.status), c.detail)
    console.print(table)

    if failures:
        console.print(f"[ovat.fail]{failures} check(s) failed.[/ovat.fail] "
                      f"Fix the red rows above.")
        raise typer.Exit(code=1)
    console.print("[ovat.ok]Everything essential looks good.[/ovat.ok]")


if __name__ == "__main__":
    app()
