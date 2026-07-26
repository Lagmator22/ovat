# ovat/cli/main.py
"""Layer 1 (entry point): the ovat command line, one YAML + one command.

This is the face of the whole toolkit. pyproject.toml points the `ovat` command
here (ovat = "ovat.cli.main:app"). typer turns each function below into a
subcommand automatically, using the type hints to parse arguments, so
`def run(config: str, ...)` becomes `ovat run <config> ...` for free.

The headline command is `run`: load a workflow YAML, build the agent, ask it
my question, print the answer. That single line is the midterm demo.
"""
import os

import typer

from ovat.agent.factory import build_agent
from ovat.cli.ui import console, esc
from ovat.config.workflow import load_workflow

# Every command prints through the ONE themed console, so the whole CLI wears
# the brand palette. `rich.print` (the old import) used a default console that
# ignored the theme; only doctor looked like OVAT, the rest looked stock.
rprint = console.print

app = typer.Typer(
    help="OVAT: run an OpenVINO agent from one YAML + one command.",
    add_completion=False,
    invoke_without_command=True,   # so bare `ovat` can open the TUI
)


@app.callback()
def _entry(ctx: typer.Context):
    """Open the TUI when `ovat` is run with no subcommand.

    typer runs this before any command. If the user typed a subcommand I get out
    of the way; if they typed just `ovat`, I launch the full-screen launcher.
    """
    if ctx.invoked_subcommand is not None:
        return
    # Recursion guard: the TUI stamps OVAT_TUI=1 into every child's env (the
    # same trick tmux uses with $TMUX). Typing `ovat` INSIDE the TUI used to
    # start a second TUI in a piped subprocess; escape codes as garbage, the
    # command slot wedged. Now it gets a hint instead, before Textual even
    # gets imported.
    import os
    if os.environ.get("OVAT_TUI"):
        rprint("[yellow]You are already inside the OVAT TUI.[/yellow] "
               "Type a subcommand instead (e.g. [bold]ovat doctor[/bold]), "
               "or /exit to leave.")
        raise typer.Exit()
    try:
        from ovat.cli.tui import run_tui
    except ImportError:
        # Textual is optional. If it is missing, fall back to the help text
        # instead of crashing, and point the user at the install.
        rprint("[yellow]The TUI needs Textual.[/yellow] Install it with "
               "[bold]pip install 'ovat\\[tui]'[/bold], or use a subcommand "
               "like [bold]ovat doctor[/bold]. See [bold]ovat --help[/bold].")
        raise typer.Exit()
    run_tui()


# The starter workflow written by `ovat init` (and by the TUI's /init shortcut,
# which runs `ovat init`). Kept here next to the command that writes it.
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

# RAG for the search_docs tool. Run `ovat index <folder> workflow.yml` first,
# then ask questions. Swap a provider string to change a backend.
rag:
  embeddings:
    provider: genai
    model: models/bge-small-en-v1.5
    device: CPU
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
        # Tool names can come from an MCP server, so they are data too.
        rprint(f"[green]Built agent[/green]  model={esc(cfg.model.name)}  "
               f"tools={esc(list(agent.tools))}  "
               f"max_iterations={agent.max_iterations}")
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
        rprint(f"[red]Error talking to OVMS at {esc(cfg.model.ovms_url)}[/red]: "
               f"{esc(exc)}")
        raise typer.Exit(code=1)
    # esc() stops the answer being read as markup; highlight=False stops rich
    # RE-styling it afterwards. Its highlighter treats plain text as a python
    # repr, so it bolds every bracket and colours bare numbers, which puts
    # escape codes INSIDE the model's words: bad to read, and worse to pipe.
    rprint(esc(answer), highlight=False)

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
    rprint(f"[dim]trace written to[/dim] {esc(path)}")


def resolve_chat_model(model_path: str | None) -> str:
    """Pick and sanity-check the local text LLM `chat` should load.

    Two jobs, both learned from real users:
    1. No path given? Auto-discover: scan OVAT_MODELS / ./models / ~/models
       and pick an instruct-tuned text LLM.
    2. Path given (or found)? IDENTIFY it before loading. Pointing chat at a
       vision model used to explode at generate time with a C++ traceback
       about tensor ports; now it is one plain sentence and a suggestion.
    Raises typer.Exit(1) after printing guidance when nothing usable exists.
    """
    from ovat.core.model_scout import find_models, identify_model, pick_chat_llm

    if model_path is None:
        choice, llms = pick_chat_llm()
        if choice is None:
            rprint("[red]No local text LLM found.[/red] I scanned OVAT_MODELS, "
                   "./models and ~/models.")
            others = find_models()
            if others:
                rprint("[dim]I did find these (wrong kind for chat):[/dim]")
                for m in others:
                    rprint(f"  [ovat.dim]{esc(m['name'])}  "
                           f"({esc(m['kind'])})[/ovat.dim]")
            rprint("Fix: pass [bold]--model-path <folder>[/bold], or set "
                   "[bold]OVAT_MODELS[/bold] to the folder that holds your "
                   "OpenVINO models.")
            raise typer.Exit(code=1)
        rprint(f"[dim]auto-detected local LLM:[/dim] "
               f"[bold]{esc(choice['name'])}[/bold]"
               f"  [dim]({esc(choice['path'])})[/dim]")
        if len(llms) > 1:
            names = ", ".join(esc(m["name"]) for m in llms if m is not choice)
            rprint(f"[dim]also available: {names}; choose with --model-path[/dim]")
        return choice["path"]

    kind, why = identify_model(model_path)
    if kind in ("llm", "unknown"):        # unknown = benefit of the doubt
        return model_path
    rprint(f"[red]{esc(os.path.basename(model_path.rstrip('/')))} is not a text "
           f"LLM[/red] [dim]({esc(why)})[/dim]; chat needs a text model.")
    _, llms = pick_chat_llm()
    if llms:
        rprint("[dim]Text LLMs found on this machine:[/dim]")
        for m in llms:
            rprint(f"  [bold]{esc(m['name'])}[/bold]  "
                   f"[ovat.dim]{esc(m['path'])}[/ovat.dim]")
    raise typer.Exit(code=1)


@app.command()
def chat(
    config: str = typer.Argument(..., help="Workflow YAML (uses its rag: section)."),
    input: str = typer.Option(..., "--input", "-i", help="Your question."),
    model_path: str = typer.Option(None, "--model-path", "-m",
                                   help="Local OpenVINO text-LLM folder. Omit to "
                                        "auto-detect (OVAT_MODELS, ./models, ~/models)."),
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

    # Resolve + identify BEFORE any heavy loading, so a wrong model kind or a
    # missing model is a one-second answer, not a 30s load then a traceback.
    model_path = resolve_chat_model(model_path)

    try:
        retriever = build_rag(cfg)
    except Exception as exc:
        rprint(f"[red]Could not build the retriever:[/red] {esc(exc)}")
        raise typer.Exit(code=1)
    try:
        llm = GenAILLMProvider(model_path, device=device, max_new_tokens=max_tokens)
    except Exception as exc:
        rprint(f"[red]Could not load the local model at "
               f"{esc(model_path)}:[/red] {esc(exc)}")
        raise typer.Exit(code=1)

    # finally: the retriever owns a SQLite connection; close it even if the
    # model call raises, so the index file is always flushed and unlocked.
    try:
        answer, sources = rag_chat(retriever, llm, input, top_k=top_k,
                                   system_prompt=cfg.agent.system_prompt)
    finally:
        retriever.close()
    rprint(esc(answer.strip()), highlight=False)   # see run(): printed as written
    if sources:
        rprint("\n[dim]sources:[/dim] " + ", ".join(esc(s) for s in sources))


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
        rprint(f"[red]Could not build the embedder/retriever:[/red] {esc(exc)}")
        rprint("[yellow]Tip:[/yellow] make sure the embeddings model in "
               f"[bold]{esc(cfg.rag.embeddings.model)}[/bold] exists on disk.")
        raise typer.Exit(code=1)

    rprint(f"[green]Indexing[/green] {esc(folder)} -> "
           f"{esc(cfg.rag.retriever.db_path)} ...")
    try:
        summary = index_folder(
            folder, retriever,
            size=cfg.rag.chunk.size, overlap=cfg.rag.chunk.overlap,
        )
    except FileNotFoundError as exc:
        rprint(f"[red]{esc(exc)}[/red]")
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
        rprint(f"[red]Refusing to overwrite existing file:[/red] {esc(path)}")
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
    rprint(f"[green]Wrote starter workflow to[/green] {esc(path)}  "
           f"[dim](model device: {esc(llm_device)}, detected)[/dim]")


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
    rprint(f"[dim]using ovms from {esc(how)}: {esc(binary)}[/dim]")
    mgr = ModelManager(binary)
    try:
        if action == "list":
            # Both of these are `ovms` subprocess output: data, not markup.
            for name in mgr.list_models():
                rprint(esc(name))
        elif action == "pull":
            if not source_model:
                rprint("[red]pull needs --source-model[/red]")
                raise typer.Exit(code=1)
            rprint(esc(mgr.pull(source_model)))
        else:
            rprint(f"[red]Unknown action '{esc(action)}'. Use list or pull.[/red]")
            raise typer.Exit(code=1)
    except FileNotFoundError:
        _ovms_not_found(how)
        raise typer.Exit(code=1)


def _ovms_not_found(how: str) -> None:
    """One consistent, ACTIONABLE message wherever ovms cannot be found."""
    rprint(f"[red]Could not find the ovms binary[/red] [dim]({esc(how)})[/dim]")
    rprint("Point OVAT at it one of these ways:")
    rprint("  1. workflow.yml →  [bold]model.ovms_binary: C:\\Users\\you\\ovms_windows[/bold]")
    rprint("  2. env var      →  [bold]set OVAT_OVMS=C:\\Users\\you\\ovms_windows[/bold]  "
           "(or export on Linux)")
    rprint("  3. classic      →  add the OVMS folder to PATH")
    rprint("[dim]macOS note: OVMS does not run on macOS at all; use 'ovat chat' "
           "locally, or serve from an AI PC / Linux box.[/dim]")


@app.command()
def serve(
    config: str = typer.Argument(..., help="Workflow YAML whose model OVMS should serve."),
    stop: bool = typer.Option(False, "--stop",
                              help="Stop the OVMS started earlier by 'ovat serve'."),
):
    """Start OVMS in the background (or stop it again with --stop). AI PC only.

    Heads up: serve returns once OVMS is READY and leaves it running in the
    background; that is the point, so `ovat run` can talk to it. The pid is
    recorded in ovms.pid; `ovat serve <config> --stop` shuts it down cleanly.
    """
    from ovat.core.model_server import ModelServer, stop_from_pidfile
    from ovat.core.ovms_locator import find_ovms

    if stop:
        # Stopping needs no config parsing at all; just the recorded pid.
        rprint(esc(stop_from_pidfile()))
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
    rprint(f"[green]Starting OVMS[/green] for {esc(cfg.model.name)} on "
           f"{esc(cfg.model.device)} [dim](binary via {esc(how)})[/dim] ...")
    try:
        server.start()
    except FileNotFoundError:
        _ovms_not_found(how)
        raise typer.Exit(code=1)
    if server.wait_until_ready():
        rprint(f"[green]OVMS is ready[/green] at {esc(server.base_url)}  "
               f"[dim](pid {server.process.pid}, "
               f"logs in {esc(server.log_path)})[/dim]")
        rprint(f"[dim]It keeps running in the background. Stop it with:[/dim] "
               f"ovat serve {esc(config)} --stop")
    else:
        rprint("[red]OVMS did not become ready in time.[/red] "
               f"See {esc(server.log_path)} for the reason.")
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
    import platform
    import sys

    from rich import box
    from rich.table import Table
    from rich.text import Text

    from ovat.cli import diagnostics
    from ovat.cli.ui import console, status_text, wordmark

    # The big sign, like the TUI launcher, but this one says what it is.
    console.print(wordmark("DOCTOR"))
    console.print("[ovat.brand]⚕ OVAT doctor[/ovat.brand]"
                  "[ovat.dim]  ·  environment & workflow diagnostics[/ovat.dim]")
    os_name = {"darwin": "macOS", "win32": "Windows"}.get(
        sys.platform, platform.system())
    console.print(f"[ovat.dim]{os_name} {platform.machine()}  ·  "
                  f"Python {sys.version.split()[0]}"
                  + (f"  ·  {esc(config)}" if config else "") + "[/ovat.dim]\n")

    checks = diagnostics.run_checks(config)

    # Row names take the status colour so the eye lands on trouble first.
    _name_style = {"ok": "ovat.cyan", "warn": "ovat.warn", "fail": "ovat.fail"}
    table = Table(header_style="ovat.header", border_style="ovat.blue",
                  box=box.ROUNDED, expand=False, pad_edge=True)
    table.add_column("Check", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail", style="ovat.dim", max_width=76)
    counts = {"ok": 0, "warn": 0, "fail": 0}
    for c in checks:
        counts[c.status] = counts.get(c.status, 0) + 1
        # c.detail carries config values, paths and exception text, so it is
        # data. The column's style still applies to the escaped string.
        table.add_row(Text(c.name, style=_name_style.get(c.status, "ovat.cyan")),
                      status_text(c.status), esc(c.detail))
    console.print(table)

    summary = Text()
    summary.append(f"✓ {counts['ok']} ok", style="ovat.ok")
    if counts["warn"]:
        summary.append("  ·  ", style="ovat.dim")
        summary.append(f"! {counts['warn']} warn", style="ovat.warn")
    if counts["fail"]:
        summary.append("  ·  ", style="ovat.dim")
        summary.append(f"✗ {counts['fail']} fail", style="ovat.fail")
    console.print(summary)

    if counts["fail"]:
        console.print("[ovat.fail]Fix the red rows above; those block "
                      "features.[/ovat.fail]")
        raise typer.Exit(code=1)
    if counts["warn"]:
        console.print("[ovat.dim]Yellow rows are heads-ups, not blockers; "
                      "each says what to do about it.[/ovat.dim]")
    else:
        console.print("[ovat.ok]All clear.[/ovat.ok]")
    if not config:
        console.print("[ovat.dim]Tip: 'ovat doctor workflow.yml' also "
                      "validates a config.[/ovat.dim]")


@app.command()
def tui():
    """Open the full-screen OVAT launcher (same as running `ovat` with no args)."""
    from ovat.cli.tui import run_tui
    run_tui()


if __name__ == "__main__":
    app()
