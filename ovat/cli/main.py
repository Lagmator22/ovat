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

from ovat.agent.factory import build_agent
from ovat.cli.ui import console
from ovat.config.workflow import load_workflow

# Every command prints through the ONE themed console, so the whole CLI wears
# the brand palette. `rich.print` (the old import) used a default console that
# ignored the theme — only doctor looked like OVAT, the rest looked stock.
rprint = console.print

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
  # Where ovms lives if it is NOT on PATH (file or folder), e.g. on Windows:
  # ovms_binary: C:\\Users\\you\\ovms_windows

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
    trace: str = typer.Option(None, "--trace",
                              help="Write a JSON run trace (tokens, latency, "
                                   "tool calls, peak memory) to this file."),
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

    # Show which engine is actually running, so it is visible in a demo: the
    # agent.type in the YAML is what picks it (native loop vs LangChain).
    engine = "LangChain (react)" if cfg.agent.type == "react" else "native loop (loop.py)"
    rprint(f"[dim]engine:[/dim] [bold]{engine}[/bold]")

    # Step 3: actually run. This needs a live OVMS server to answer.
    try:
        answer = agent.run(input)
    except Exception as exc:
        rprint(f"[red]Error talking to OVMS at {cfg.model.ovms_url}[/red]: {exc}")
        raise typer.Exit(code=1)
    rprint(answer)

    if trace:
        _write_trace(trace, cfg, agent)


def _write_trace(path: str, cfg, agent) -> None:
    """Dump the run trace (Layer 7) as JSON: what the run cost, measured.

    The native loop fills agent.last_trace as it works; peak RSS comes from
    psutil so the proposal's memory-budget criterion (<8 GB) is a number in a
    file, not a claim. The react engine does not expose per-turn data yet, so
    its trace says so honestly instead of writing empty numbers.
    """
    import json

    trace_data = getattr(agent, "last_trace", None) or {
        "engine": "react",
        "note": "per-turn tracing is only wired for the native loop so far",
    }
    trace_data = dict(trace_data)                  # never mutate the agent's copy
    trace_data["model"] = cfg.model.name
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
        trace_data["peak_rss_mb"] = round(rss / (1024 * 1024), 1)
    except ImportError:
        trace_data["peak_rss_mb"] = None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)
    rprint(f"[dim]trace written to[/dim] {path}")


@app.command()
def chat(
    config: str = typer.Argument(..., help="Workflow YAML (uses its rag: section)."),
    input: str = typer.Option(..., "--input", "-i", help="Your question."),
    model_path: str = typer.Option(..., "--model-path", "-m",
                                   help="Path to a local OpenVINO LLM folder, e.g. Llama-3.2-3B."),
    device: str = typer.Option("CPU", "--device", help="CPU, or GPU/NPU on the AI PC."),
    top_k: int = typer.Option(4, "--top-k", help="How many chunks to retrieve."),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Answer length cap."),
):
    """Chat with your documents using a LOCAL OpenVINO model (no OVMS needed).

    The macOS path: this embeds your question, retrieves the closest chunks from
    the index you built with `ovat index`, and answers with a local model, citing
    its sources. It is real retrieval-augmented generation; it does not tool-call
    (that needs OVMS) but always retrieves then answers. Index first, then ask.
    """
    from ovat.agent.factory import build_rag
    from ovat.agent.rag_chat import rag_chat
    from ovat.providers.llm_genai import GenAILLMProvider

    cfg = load_workflow(config)
    if cfg.rag is None:
        rprint("[red]This workflow has no [bold]rag:[/bold] section to chat against.[/red]")
        raise typer.Exit(code=1)

    try:
        retriever = build_rag(cfg)
    except Exception as exc:
        rprint(f"[red]Could not build the retriever:[/red] {exc}")
        raise typer.Exit(code=1)
    try:
        llm = GenAILLMProvider(model_path, device=device, max_new_tokens=max_tokens)
    except Exception as exc:
        rprint(f"[red]Could not load the local model at {model_path}:[/red] {exc}")
        raise typer.Exit(code=1)

    # finally: the retriever owns a SQLite connection; close it even if the
    # model call raises, so the index file is always flushed and unlocked.
    try:
        answer, sources = rag_chat(retriever, llm, input, top_k=top_k,
                                   system_prompt=cfg.agent.system_prompt)
    finally:
        retriever.close()
    rprint(answer.strip())
    if sources:
        rprint("\n[dim]sources:[/dim] " + ", ".join(sources))


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
    finally:
        # Close the vector store so every chunk is flushed to the .db file and
        # its lock is released, even when indexing fails halfway.
        retriever.close()
    rprint(f"[green]Indexed[/green] {summary['chunks']} chunks "
           f"from {summary['files']} files.")


@app.command()
def init(
    path: str = typer.Argument("workflow.yml", help="Where to write the starter YAML."),
):
    """Write a starter workflow.yml tuned to THIS machine's hardware."""
    import os
    if os.path.exists(path):
        rprint(f"[red]Refusing to overwrite existing file:[/red] {path}")
        raise typer.Exit(code=1)
    # DeviceManager picks the LLM device for the hardware we are on: GPU on
    # an AI PC, CPU on a laptop/Mac. The starter file should run where it
    # was created, not assume a GPU that may not exist.
    try:
        from ovat.core.device_manager import DeviceManager
        llm_device = DeviceManager().get_llm_device()
    except Exception:
        llm_device = "CPU"                       # the universal fallback
    starter = _STARTER_YAML.replace("device: GPU", f"device: {llm_device}", 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(starter)
    rprint(f"[green]Wrote starter workflow to[/green] {path}  "
           f"[dim](model device: {llm_device}, detected)[/dim]")


@app.command()
def models(
    action: str = typer.Argument("list", help="list or pull"),
    source_model: str = typer.Option(None, "--source-model",
                                     help="Hugging Face id to pull, e.g. OpenVINO/Qwen3-8B-int4-ov."),
):
    """List or pull OVMS models (needs the ovms binary, so runs on the AI PC)."""
    from ovat.core.model_manager import ModelManager
    from ovat.core.ovms_locator import find_ovms

    binary, how = find_ovms()
    if binary is None:
        _ovms_not_found(how)
        raise typer.Exit(code=1)
    rprint(f"[dim]using ovms from {how}: {binary}[/dim]")
    mgr = ModelManager(binary)
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
        _ovms_not_found(how)
        raise typer.Exit(code=1)


def _ovms_not_found(how: str) -> None:
    """One consistent, ACTIONABLE message wherever ovms cannot be found."""
    rprint(f"[red]Could not find the ovms binary[/red] [dim]({how})[/dim]")
    rprint("Point OVAT at it one of these ways:")
    rprint("  1. workflow.yml →  [bold]model.ovms_binary: C:\\Users\\you\\ovms_windows[/bold]")
    rprint("  2. env var      →  [bold]set OVAT_OVMS=C:\\Users\\you\\ovms_windows[/bold]  "
           "(or export on Linux)")
    rprint("  3. classic      →  add the OVMS folder to PATH")
    rprint("[dim]macOS note: OVMS does not run on macOS at all — use 'ovat chat' "
           "locally, or serve from an AI PC / Linux box.[/dim]")


@app.command()
def serve(
    config: str = typer.Argument(..., help="Workflow YAML whose model OVMS should serve."),
    stop: bool = typer.Option(False, "--stop",
                              help="Stop the OVMS started earlier by 'ovat serve'."),
):
    """Start OVMS in the background (or stop it again with --stop). AI PC only.

    Heads up: serve returns once OVMS is READY and leaves it running in the
    background — that is the point, so `ovat run` can talk to it. The pid is
    recorded in ovms.pid; `ovat serve <config> --stop` shuts it down cleanly.
    """
    from ovat.core.model_server import ModelServer, stop_from_pidfile
    from ovat.core.ovms_locator import find_ovms

    if stop:
        # Stopping needs no config parsing at all — just the recorded pid.
        rprint(stop_from_pidfile())
        return

    cfg = load_workflow(config)
    # Resolve the binary FIRST (config field → OVAT_OVMS env → PATH → known
    # folders), so a setupvars.bat-style install just works with no PATH edit.
    binary, how = find_ovms(cfg.model.ovms_binary)
    if binary is None:
        _ovms_not_found(how)
        raise typer.Exit(code=1)
    server = ModelServer(
        model_name=cfg.model.name,
        source_model=cfg.model.source_model,
        model_repository_path=cfg.model.model_repository_path,
        device=cfg.model.device,
        tool_parser=cfg.model.tool_parser,
        reasoning_parser=cfg.model.reasoning_parser,
        enable_prefix_caching=cfg.model.enable_prefix_caching,
        binary=binary,
    )
    rprint(f"[green]Starting OVMS[/green] for {cfg.model.name} on {cfg.model.device} "
           f"[dim](binary via {how})[/dim] ...")
    try:
        server.start()
    except FileNotFoundError:
        _ovms_not_found(how)
        raise typer.Exit(code=1)
    if server.wait_until_ready():
        rprint(f"[green]OVMS is ready[/green] at {server.base_url}  "
               f"[dim](pid {server.process.pid}, logs in {server.log_path})[/dim]")
        rprint(f"[dim]It keeps running in the background. Stop it with:[/dim] "
               f"ovat serve {config} --stop")
    else:
        rprint("[red]OVMS did not become ready in time.[/red] "
               f"See {server.log_path} for the reason.")
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
