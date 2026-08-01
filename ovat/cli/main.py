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
from rich import box
from rich.table import Table
from rich.text import Text
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn,
                           TimeRemainingColumn)

from ovat.agent.factory import build_agent
from ovat.cli import ui
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
  # Qwen3.5-4B is a UNIFIED model: text, vision and tool calling in one
  # export, so the same file drives the RAG, ReAct and image examples.
  # 3.5 GB to download, about 5 GB of RAM. Smaller box? Swap both lines
  # below for Qwen3.5-0.8B-int4-ov (0.9 GB, ~2 GB RAM). Bigger box and want
  # the strongest text answers? Qwen3-8B-int4-ov (4.9 GB) still works.
  name: Qwen3.5-4B-int4-ov
  device: GPU
  ovms_url: http://localhost:8000/v3
  # auto = let OVMS read the model's chat template and pick the parser.
  # Prefer this over naming one: an explicit parser OVERRIDES OVMS's own
  # detection, and the right answer differs per family (Qwen3.5 emits
  # qwen3coder-shaped calls, Qwen3 hermes3-shaped ones). Name one only to
  # override, e.g. tool_parser: hermes3
  tool_parser: auto
  # Only used by `ovat serve` to start OVMS and locate the model:
  source_model: OpenVINO/Qwen3.5-4B-int4-ov
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

# RAG for the search_docs tool: OFF until you switch it on, and deliberately.
#
# An active rag: block here would name an embedding model that is not on a new
# machine, so `ovat index` -- the very next command in the quickstart -- died
# with a C++ assertion out of OpenVINO's frontend. Nothing before it failed,
# so the config looked fine right up until it did not. Without this block
# search_docs answers in its documented stub mode instead, and every command
# in the quickstart works on a fresh clone with no downloads.
#
# To switch on real vector search, export the embedder ONCE (about 130 MB).
# optimum-cli is NOT installed by default; it lives in the `convert` extra:
#
#   pip install "ovat[convert]"
#   optimum-cli export openvino --model BAAI/bge-small-en-v1.5 \\
#       --task feature-extraction models/bge-small-en-v1.5
#
# then uncomment the block below and run:
#
#   ovat index <your-folder> workflow.yml
#
# Every value is a swappable string: provider: ovms embeds on the server
# instead of locally, and db_path can live anywhere.
#
# rag:
#   embeddings:
#     provider: genai              # genai (local) | ovms (server /v3)
#     model: models/bge-small-en-v1.5
#     device: CPU                  # CPU or NPU
#     dim: 384                     # bge-small emits 384 numbers per chunk
#   retriever:
#     provider: sqlite-vec
#     db_path: ovat_index.db
#   chunk:
#     size: 512                    # characters per chunk
#     overlap: 64                  # characters shared with the next chunk
"""


@app.command()
def run(
    config: str = typer.Argument(..., help="Path to a workflow YAML."),
    input: str = typer.Option(None, "--input", "-i",
                              help="Your question for the agent. Not needed "
                                   "with --dry-run."),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="Build the agent and show it, but do not call the model."),
    trace: str = typer.Option(None, "--trace",
                              help="Write a JSON run trace (tokens, latency, "
                                   "tool calls, peak memory) to this file."),
    telemetry: str = typer.Option(None, "--telemetry",
                                  help="Stream live telemetry to this file as "
                                       "JSON Lines (process memory, and Intel "
                                       "GPU/NPU when available)."),
):
    """Run the agent described by CONFIG against your input.

    --input is REQUIRED for a real run and IGNORED by --dry-run, which only
    builds the agent. Demanding a question in order to not ask one made the
    commonest smoke test (`ovat run workflow.yml --dry-run`) fail with
    "Missing option --input", which reads as the command being broken.
    """
    if input is None and not dry_run:
        rprint("[red]Missing --input.[/red] A real run needs a question:")
        # soft_wrap so a long config path is NOT broken across lines. A
        # suggested command that has to be un-wrapped by hand before it can
        # be pasted is not a suggestion, it is a puzzle.
        rprint(f"  [bold]ovat run {esc(config)} --input \"your question\""
               f"[/bold]", soft_wrap=True)
        rprint("[dim]Or use --dry-run to just build the agent and show it.[/dim]")
        raise typer.Exit(code=1)
    # Step 1: YAML -> validated config. A bad file fails loudly right here.
    cfg = _load_config(config)
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

    # Show which engine is actually running, so it is visible in a demo. The
    # name comes from the CONFIG; see ENGINE_LABELS for why this is not a
    # conditional expression any more.
    rprint(f"[dim]engine:[/dim] [bold]{esc(_engine_label(cfg.agent.type))}[/bold]")

    # Step 3: actually run. This needs a live OVMS server to answer.
    # The memory sampler runs for the whole call: see _write_trace for why a
    # single reading afterwards is not a peak.
    from ovat.bench import _PeakMemory

    # Live telemetry alongside the run, if asked for. The collector owns its
    # own thread, so a stalled hardware collector cannot slow the agent down.
    collector = _start_telemetry(telemetry, agent) if telemetry else None
    memory = _PeakMemory()
    try:
        with memory:
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

    if collector is not None:
        collector.stop()
        rprint(f"[dim]telemetry written to[/dim] {esc(telemetry)}")
    if trace:
        _write_trace(trace, cfg, agent, peak_rss_mb=memory.peak_mb)


def _exit_is_a_folder(path: str):
    """One wording for "you gave me a directory", reached from two excepts.

    POSIX gets here via IsADirectoryError, Windows via PermissionError. The
    user made one mistake, so they get one sentence.
    """
    rprint(f"[red]That is a folder, not a workflow file:[/red] {esc(path)}")
    raise typer.Exit(code=1)


# How each engine is announced on screen. Presentation only: the list of
# engines that actually EXIST is factory.AGENT_TYPES, and this must not become
# a second copy of it.
#
# This line used to be
#     "LangChain (react)" if cfg.agent.type == "react" else "native loop"
# which was true while there were two engines and became a lie the moment
# there were four: a llamaindex run announced itself as the native loop. The
# trace had the identical defect and was fixed; this, the line a demo actually
# points at, was not, so the screen and the JSON disagreed about one run.
#
# The .get() fallback is the part that matters. An engine added to the factory
# and forgotten here prints its own bare name, which is terse but TRUE. There
# is no branch left that can attribute a run to the wrong engine.
ENGINE_LABELS = {
    "native": "native loop (loop.py)",
    "react": "LangChain (react)",
    "llamaindex": "LlamaIndex (llamaindex)",
    "openai-agents": "OpenAI Agents SDK (openai-agents)",
}


def _engine_label(agent_type: str) -> str:
    """Human name for an engine, falling back to the configured name."""
    return ENGINE_LABELS.get(agent_type, agent_type)


def _load_config(path: str):
    """load_workflow, but a bad file is a SENTENCE rather than a traceback.

    Every command starts by reading a workflow, so every command inherited
    the raw exception: a mistyped name (workflow.yaml for workflow.yml) put a
    twenty-line rich traceback on screen with the useful part buried in it.
    The project rule is that failures reach the user as something they can
    act on, and this is the single busiest place that rule applies.

    Three distinct failures, three distinct sentences, because the fix is
    different for each: the file is not there, the YAML does not parse, or it
    parses but does not match the schema.
    """
    import yaml
    from pydantic import ValidationError

    try:
        return load_workflow(path)
    except FileNotFoundError:
        rprint(f"[red]No such workflow file:[/red] {esc(path)}")
        # A path with spaces and no YAML extension is almost never a mistyped
        # filename; it is a QUESTION that landed in the config slot because
        # --input ate the config. Saying "run ovat init" there is actively
        # misleading, so name the real mistake and show the right order.
        looks_like_a_question = " " in path.strip() and not path.endswith(
            (".yml", ".yaml"))
        if looks_like_a_question:
            rprint("[yellow]That looks like a question, not a file.[/yellow] "
                   "The config comes FIRST and --input takes the question:")
            rprint(f"  [bold]ovat run <workflow.yml> --input "
                   f"\"{esc(path)}\"[/bold]")
        elif path.endswith(".yaml"):
            rprint("[yellow]Tip:[/yellow] OVAT's examples use the "
                   "[bold].yml[/bold] spelling. Try "
                   f"[bold]{esc(path[:-len('.yaml')] + '.yml')}[/bold]")
        else:
            rprint("[yellow]Tip:[/yellow] run [bold]ovat init[/bold] to "
                   "write a starter workflow.yml here.")
        raise typer.Exit(code=1)
    except IsADirectoryError:
        _exit_is_a_folder(path)
    except PermissionError:
        # Windows does NOT raise IsADirectoryError when open() is handed a
        # directory: it raises PermissionError (EACCES). The branch above is
        # right on POSIX and never fires on the AI PC, which is the primary
        # target, so `ovat run some_folder` printed the raw traceback there
        # that rule 6 exists to prevent. Ask the filesystem which case it is
        # rather than trusting the exception class to mean the same thing on
        # both platforms.
        if os.path.isdir(path):
            _exit_is_a_folder(path)
        rprint(f"[red]Cannot read {esc(path)}:[/red] permission denied.")
        raise typer.Exit(code=1)
    except yaml.YAMLError as exc:
        rprint(f"[red]{esc(path)} is not valid YAML.[/red]")
        # mark tells you the line and column; without it the user is hunting
        # a stray bracket by eye.
        mark = getattr(exc, "problem_mark", None)
        if mark is not None:
            rprint(f"[dim]line {mark.line + 1}, column {mark.column + 1}"
                   f"[/dim]  {esc(getattr(exc, 'problem', '') or '')}")
        raise typer.Exit(code=1)
    except ValidationError as exc:
        rprint(f"[red]{esc(path)} does not match the workflow schema.[/red]")
        for error in exc.errors():
            where = ".".join(str(part) for part in error["loc"]) or "(root)"
            rprint(f"  [bold]{esc(where)}[/bold]: {esc(error['msg'])}")
        # Unknown keys are the common case and the message for them is
        # otherwise just "Extra inputs are not permitted", which does not say
        # that the strictness is deliberate.
        if any(e["type"] == "extra_forbidden" for e in exc.errors()):
            rprint("[dim]Unknown keys are rejected on purpose, so a typo is "
                   "an error rather than a setting that silently does "
                   "nothing.[/dim]")
        raise typer.Exit(code=1)


def _start_telemetry(path: str, agent):
    """Begin streaming telemetry to `path` as JSON Lines, and return the
    collector so the caller can stop it.

    JSON Lines rather than one array: a run that is killed half-way still
    leaves a readable file, and `tail -f` works while it is going.
    """
    from ovat.telemetry.collector import Collector
    from ovat.telemetry.sinks import JSONFileSink
    from ovat.telemetry.sources import (AgentTraceSource, IntelHardwareSource,
                                        ProcessMemorySource)

    collector = Collector(
        [ProcessMemorySource(), AgentTraceSource(agent),
         IntelHardwareSource(getattr(agent, "ut_binary", None))],
        JSONFileSink(path), interval_s=0.5)
    # Say which sources are NOT running here. A file of process memory alone,
    # with no note that the hardware source was skipped, reads as a machine
    # with an idle NPU rather than one that cannot measure it.
    for name, reason in collector.unavailable.items():
        rprint(f"[dim]telemetry: {esc(name)} unavailable ({esc(reason)})[/dim]")
    collector.start()
    return collector


def _brief_error(message: str | None, limit: int = 34) -> str:
    """The most ACTIONABLE fragment of an error, short enough for a cell.

    "pip install 'ovat[llamaindex]'" tells the reader what to do; the words
    "RuntimeError" tell them nothing they cannot already see from the colour.
    """
    if not message:
        return "failed"
    # An install hint is the whole point of the message when there is one.
    if "pip install" in message:
        hint = message[message.index("pip install"):]
        return hint if len(hint) <= limit else hint[:limit - 1] + "\u2026"
    # Otherwise drop the exception class and keep what it said.
    _, _, detail = message.partition(": ")
    text = (detail or message).strip()
    return text if len(text) <= limit else text[:limit - 1] + "\u2026"


def _write_trace(path: str, cfg, agent, peak_rss_mb=None) -> None:
    """Dump the run trace (Layer 7) as JSON: what the run cost, measured.

    The native loop fills agent.last_trace as it works. The framework engines
    own their own request loops and do not hand per-turn data back, so their
    trace says so rather than writing empty numbers.

    The engine name comes from the CONFIG, not from a literal. It used to be
    hardcoded "react", which was true when react was the only framework
    engine and became a lie the moment there were three: a llamaindex run
    wrote a trace file claiming it was react. A measurement that misreports
    what produced it is worse than no measurement.

    peak_rss_mb is passed IN because it has to be sampled while the run is
    happening. Reading it here, after the run, misses the peak entirely:
    Python has usually freed the big allocations by then, and this number
    exists for the proposal's <8 GB memory criterion.
    """
    import json

    trace_data = getattr(agent, "last_trace", None) or {
        "engine": cfg.agent.type,
        "note": ("per-turn tracing is only wired for the native loop; this "
                 "engine owns its own request loop and does not report "
                 "token usage back"),
    }
    trace_data = dict(trace_data)                  # never mutate the agent's copy
    trace_data["model"] = cfg.model.name
    trace_data["peak_rss_mb"] = peak_rss_mb
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
    # "unified" is a text LLM that also takes images (Qwen3.5); refusing it
    # would reject the model the quickstart itself recommends.
    if kind in ("llm", "unified", "unknown"):   # unknown = benefit of the doubt
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
    max_tokens: int = typer.Option(256, "--max-tokens",
                                   help="Answer length cap; 0 means no cap "
                                        "(generate until the model stops)."),
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

    cfg = _load_config(config)
    if cfg.rag is None:
        rprint("[red]This workflow has no [bold]rag:[/bold] section to chat against.[/red]")
        raise typer.Exit(code=1)
    if max_tokens < 0:
        rprint("[red]--max-tokens cannot be negative.[/red] Use a positive cap, "
               "or 0 for no cap.")
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
        # 0 -> None: the provider reads None as "no cap".
        llm = GenAILLMProvider(model_path, device=device,
                               max_new_tokens=max_tokens or None)
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
    from ovat.rag.indexer import index_folder, iter_text_files

    cfg = _load_config(config)
    if cfg.rag is None:
        # `ovat init` now ships the rag: block COMMENTED OUT, so this is the
        # normal path for a new user rather than a rare mistake: they followed
        # the quickstart, got here, and this message is the only thing standing
        # between them and working retrieval. Naming what is absent is not
        # enough; the old wording ("Add one (embeddings + retriever)") told
        # someone who has never exported an OpenVINO model precisely nothing.
        #
        # So: say it is OFF, not BROKEN; say the tool still works meanwhile, so
        # they are not blocked; and point at the one place the instructions
        # already live rather than printing a second copy that can drift.
        rprint(f"[yellow]Vector search is off in[/yellow] {esc(config)}[yellow];"
               " nothing is broken.[/yellow]")
        rprint("[bold]search_docs still answers[/bold] in stub mode, so "
               "[cyan]ovat run[/cyan] works right now without this.")
        rprint(f"To switch on real retrieval, open {esc(config)} and follow "
               "the commented [bold]rag:[/bold] block: export the embedder "
               "once with the [cyan]optimum-cli[/cyan] line above it, "
               "uncomment, then run this command again.")
        # optimum-cli ships in an extra, so the line above is "command not
        # found" until this runs. Saying so here costs one line and saves the
        # user discovering it the hard way half way through the quickstart.
        rprint("[dim]optimum-cli comes from[/dim] [bold]pip install "
               "\"ovat\\[convert]\"[/bold][dim]; install it first.[/dim]")
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
        # A bar, not silence. Embedding a few hundred files is minutes of an
        # unmoving cursor, and there was no way to tell that from a hang.
        # transient=True so the finished bar does not outlive the run and
        # compete with the summary line for the reader's attention.
        # Count the files BEFORE starting. index_folder only reveals the total
        # with its first callback, and until then the bar reads "0/None",
        # which on a large folder is several seconds of what looks like a
        # broken bar. Walking the tree twice costs nothing next to embedding.
        file_count = len(list(iter_text_files(folder)))
        with Progress(
            SpinnerColumn(style=ui.BLUE),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style=ui.GREEN, finished_style=ui.GREEN),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total} files"),
            TimeRemainingColumn(),
            console=console, transient=True,
        ) as progress:
            task = progress.add_task("embedding", total=file_count)

            def tick(done: int, total: int, path: str) -> None:
                progress.update(task, completed=done, total=total,
                                description=os.path.basename(path)[:30])

            summary = index_folder(
                folder, retriever,
                size=cfg.rag.chunk.size, overlap=cfg.rag.chunk.overlap,
                on_progress=tick,
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
    repo: str = typer.Option("models", "--repo",
                             help="Model repository folder OVMS reads and writes."),
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
            names = mgr.list_models(repo)
            if not names:
                rprint(f"[yellow]No models in[/yellow] {esc(repo)}. "
                       f"Pull one with: ovat models pull --source-model <hf-id>")
            for name in names:
                rprint(esc(name))
        elif action == "pull":
            if not source_model:
                rprint("[red]pull needs --source-model[/red]")
                raise typer.Exit(code=1)
            rprint(esc(mgr.pull(source_model, repo)))
        else:
            rprint(f"[red]Unknown action '{esc(action)}'. Use list or pull.[/red]")
            raise typer.Exit(code=1)
    except FileNotFoundError:
        _ovms_not_found(how)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        # ovms ran but refused. Only FileNotFoundError was caught before, so
        # this escaped as a raw traceback: the "errors are for users" rule
        # applies just as much to the tool we shell out to.
        rprint(f"[red]ovms could not do that:[/red] {esc(exc)}")
        rprint(f"[dim]repository:[/dim] {esc(repo)}  "
               f"[dim]· point elsewhere with[/dim] --repo <folder>")
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

    cfg = _load_config(config)
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


@app.command()
def telemetry(
    seconds: float = typer.Option(10.0, "--seconds", "-s",
                                  help="How long to sample for. 0 runs until "
                                       "Ctrl-C."),
    interval: float = typer.Option(1.0, "--interval",
                                   help="Seconds between samples."),
    out: str = typer.Option(None, "--out",
                            help="Also stream every sample to this file as "
                                 "JSON Lines."),
    ut: str = typer.Option(None, "--ut",
                           help="Path to the Intel Unified Telemetry folder "
                                "or binary, if it was not auto-detected. "
                                "OVAT_UT does the same thing."),
    once: bool = typer.Option(False, "--once",
                              help="Print one snapshot and exit."),
):
    """Show live system, process and Intel hardware telemetry.

    The same sources the TUI's /telemetry page draws, for people who live in
    the terminal or want to pipe the numbers somewhere. `ovat run --telemetry`
    measures ONE run; this measures the machine.
    """
    import time

    from ovat.telemetry.collector import Collector
    from ovat.telemetry.sinks import FanOutSink, JSONFileSink, LiveBufferSink
    from ovat.telemetry.sources import (IntelHardwareSource,
                                        ProcessMemorySource, SystemSource)

    live = LiveBufferSink()
    sink = FanOutSink(live, JSONFileSink(out)) if out else live
    collector = Collector([SystemSource(), ProcessMemorySource(),
                           IntelHardwareSource(ut)], sink,
                          interval_s=interval)

    # Say up front what will NOT be measured here. A table of CPU alone, with
    # no note that the hardware source was skipped, reads as a machine with an
    # idle NPU rather than one that cannot see it.
    for name, reason in collector.unavailable.items():
        rprint(f"[yellow]{esc(name)}[/yellow] unavailable: [dim]{esc(reason)}"
               f"[/dim]")
    if not collector.available:
        rprint("[red]No telemetry sources work here.[/red]")
        raise typer.Exit(code=1)

    if once:
        _print_telemetry(collector.sample_once())
        return

    # rich.Live redraws ONE table in place instead of printing a new one every
    # tick. Printing scrolled a fresh table every interval, which made a
    # terminal unreadable within seconds and buried the numbers it was meant
    # to show.
    from rich.live import Live

    collector.start()
    deadline = None if seconds <= 0 else time.time() + seconds
    try:
        with Live(_telemetry_table({}), console=console,
                  refresh_per_second=4, transient=False) as display:
            while deadline is None or time.time() < deadline:
                time.sleep(interval)
                if live.samples:
                    display.update(_telemetry_table(live.samples[-1]))
    except KeyboardInterrupt:
        pass
    finally:
        collector.stop()
        if out:
            sink.close()
            rprint(f"[dim]telemetry written to[/dim] {esc(out)}")


def _print_telemetry(sample: dict) -> None:
    """One snapshot, printed once. Used by --once."""
    if sample:
        console.print(_telemetry_table(sample))


def _telemetry_table(sample: dict) -> Table:
    """One snapshot as a table, grouped by source.

    Returns the table rather than printing it, so live mode can hand the same
    renderable to rich.Live and have it redrawn in place.
    """
    table = Table(header_style=f"bold {ui.BLUE}", border_style=ui.BLUE,
                  box=box.ROUNDED)
    table.add_column("Source", no_wrap=True)
    table.add_column("Metric", no_wrap=True)
    table.add_column("Value", justify="right", no_wrap=True)
    for key in sorted(sample):
        if key == "t":
            continue
        source, _, metric = key.partition(".")
        value = sample[key]
        table.add_row(Text(source, style=ui.CYAN), Text(metric, style=ui.DIM),
                      Text(f"{value:,.1f}", style=f"bold {ui.GREEN}"))
    if not sample:
        table.add_row(Text("...", style=ui.DIM), Text("sampling", style=ui.DIM),
                      Text("", style=ui.DIM))
    return table


@app.command()
def bench(
    config: str = typer.Argument(..., help="Path to a workflow YAML."),
    input: str = typer.Option(..., "--input", "-i",
                              help="The question to run through every engine."),
    engines: str = typer.Option(
        "native,react,llamaindex,openai-agents", "--engines",
        help="Comma-separated engines to compare."),
    out: str = typer.Option(None, "--out",
                            help="Write the full report as JSON to this file."),
):
    """Run one question through several engines and compare what each costs.

    This is the AI PC profiling deliverable: the claim is "one YAML, any
    framework, the same OVMS backend", and the only way to make that claim
    mean anything is the same question through every engine against the same
    server, side by side.
    """
    import json

    from ovat.bench import benchmark

    cfg = _load_config(config)
    names = [e.strip() for e in engines.split(",") if e.strip()]
    if not names:
        rprint("[red]No engines to run.[/red] Pass --engines native,react")
        raise typer.Exit(code=1)

    rprint(f"[green]Benchmarking[/green] {esc(cfg.model.name)} at "
           f"{esc(cfg.model.ovms_url)}  [dim]({len(names)} engines)[/dim]")
    report = benchmark(cfg, input, names, config_path=config)

    table = Table(header_style=f"bold {ui.BLUE}", border_style=ui.BLUE,
                  box=box.ROUNDED)
    for column in ("Engine", "Build s", "Answer s", "Peak MB", "Prompt tok",
                   "Reply tok", "Tools"):
        table.add_column(column, no_wrap=True)
    table.add_column("Result", no_wrap=True)
    for row in report["results"]:
        # A dash, never a zero: "unknown" and "none" are different claims, and
        # only the native loop records token usage.
        def cell(key):
            value = row[key]
            return "-" if value is None else str(value)
        # The table is a SUMMARY, so the message is shortened. Keep the part
        # that says WHAT TO DO, not the exception class.
        #
        # This was the other way round and it actively misled: a missing
        # extra reads "RuntimeError: agent.type 'llamaindex' needs LlamaIndex.
        # Install it with: pip install 'ovat[llamaindex]'", and showing only
        # the class rendered a bare "RuntimeError" next to two engines that
        # worked. It looked like the engine was broken rather than simply not
        # installed, which is the opposite of what the row was trying to say.
        if row["ok"]:
            status = Text("ok", style=f"bold {ui.GREEN}")
        else:
            status = Text(_brief_error(row["error"]), style=ui.RED)
        table.add_row(Text(row["engine"], style=ui.CYAN), cell("build_s"),
                      cell("latency_s"), cell("peak_rss_mb"),
                      cell("prompt_tokens"), cell("completion_tokens"),
                      cell("tool_calls"), status)
    console.print(table)

    if any(not r["ok"] for r in report["results"]) and not out:
        rprint("[dim]Pass --out report.json for the full error text.[/dim]")
    if any(r["ok"] and r["prompt_tokens"] is None for r in report["results"]):
        rprint("[dim]Token counts come from OVMS's usage field, which only "
               "the native loop records; the frameworks own their own request "
               "loops and do not hand it back.[/dim]")
    # Peak MB is only comparable between rows because each engine now runs in
    # its OWN process. RSS is a whole-process number, so when they shared one,
    # every row inherited what the rows above it had allocated: measured on
    # the AI PC, native read 465.8 MB running first and 1155.6 MB running
    # last, against 466 MB measured alone. Reordering the list reversed the
    # conclusion, which means the old table was measuring list position.
    if report.get("isolated") and len(names) > 1:
        rprint("[dim]Each engine was measured in its own process, so Peak MB "
               "is that engine's own cost.[/dim]")

    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        rprint(f"[dim]report written to[/dim] {esc(out)}")


if __name__ == "__main__":
    app()
