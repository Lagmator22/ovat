# ovat/bench.py
"""Run one question through several engines and measure what each one costs.

This is the proposal's AI PC profiling deliverable. The claim OVAT makes is
"one YAML, any framework, the same OVMS backend", and the only way to show
that is worth anything is to run the SAME question through every engine
against the SAME server and put the numbers side by side.

Kept out of cli/main.py because it is real logic with real edge cases and
deserves its own tests, not a place inside a typer command body.

Two measurement honesty notes, both of which shape the output:

  * Token counts come from OVMS's usage field, which every engine records
    in agent.last_trace (the framework adapters read it back from their
    frameworks; see ovat/agent/usage.py). Where a server sent none, tokens
    stay explicitly unknown, never zero. A zero would read as "this engine
    used no tokens", which is false.
  * Peak RSS is SAMPLED on a background thread, not read once at the end. A
    single reading after the run misses the peak entirely, because Python has
    usually already freed the big allocations by then, and the whole point of
    the number is the proposal's <8 GB memory criterion.
"""
import threading
import time


class _PeakMemory:
    """Sample this process's RSS on a thread and keep the highest value.

    A context manager so the sampler cannot outlive the run it is measuring,
    including when that run raises.
    """

    def __init__(self, interval: float = 0.05):
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.peak_bytes = 0

    def __enter__(self):
        try:
            import psutil
        except ImportError:
            return self                 # no psutil: peak stays 0, reported None
        process = psutil.Process()
        self.peak_bytes = process.memory_info().rss

        def sample():
            while not self._stop.is_set():
                try:
                    self.peak_bytes = max(self.peak_bytes,
                                          process.memory_info().rss)
                except Exception:
                    return              # process gone; nothing to measure
                self._stop.wait(self._interval)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False

    @property
    def peak_mb(self):
        return round(self.peak_bytes / (1024 * 1024), 1) if self.peak_bytes \
            else None


def benchmark_engine(config, engine: str, question: str,
                     build_agent=None) -> dict:
    """Run `question` through one engine and return a result row.

    Never raises. A missing framework or an unreachable OVMS is a RESULT, not
    a crash: the whole point of a comparison table is that the engine which
    failed is visible next to the ones that did not.
    """
    if build_agent is None:
        from ovat.agent.factory import build_agent

    row = {"engine": engine, "ok": False, "error": None, "answer": "",
           "latency_s": None, "build_s": None, "peak_rss_mb": None,
           "prompt_tokens": None, "completion_tokens": None,
           "tool_calls": None}

    # A copy, so one engine's row cannot mutate the config the next one reads.
    run_config = config.model_copy(deep=True)
    run_config.agent.type = engine

    agent = None
    with _PeakMemory() as memory:
        build_started = time.monotonic()
        try:
            agent = build_agent(run_config)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["peak_rss_mb"] = memory.peak_mb
            return row
        row["build_s"] = round(time.monotonic() - build_started, 3)

        run_started = time.monotonic()
        try:
            row["answer"] = agent.run(question)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["latency_s"] = round(time.monotonic() - run_started, 3)
            row["peak_rss_mb"] = memory.peak_mb
            return row
        finally:
            if agent is not None:
                from ovat.agent.factory import close_agent
                close_agent(agent)
        row["latency_s"] = round(time.monotonic() - run_started, 3)

    row["peak_rss_mb"] = memory.peak_mb

    # Tokens only where the engine actually recorded them. See the module
    # docstring: absent is not zero. Read BEFORE deciding ok, because the trace
    # is what says whether this run actually did anything.
    totals = (getattr(agent, "last_trace", None) or {}).get("totals") or {}
    for key in ("prompt_tokens", "completion_tokens", "tool_calls"):
        if totals.get(key) is not None:
            row[key] = totals[key]

    # "ok" means the engine PRODUCED AN ANSWER, not merely that it did not
    # raise. An engine whose reply was swallowed by a mismatched tool_parser
    # returns cleanly having said nothing, and scoring that ok made the table
    # lie: a green row for a run that answered nothing.
    #
    # The TRACE decides this, not the answer text. A previous attempt tested
    # says_nothing(row["answer"]) and was defeated by a change made in the SAME
    # commit: the loop now turns an empty reply into an "Error: the model
    # returned no answer" string, which is not empty, so the check passed and
    # the row still scored ok. Verified on live hardware. The flags cannot be
    # fooled that way, because they record what HAPPENED, not what was printed.
    from ovat.text import looks_like_undecoded_tool_call, says_nothing

    failure = None
    if totals.get("empty_answer"):
        failure = ("the engine returned no answer (reply was empty, or only "
                   "reasoning). Usually a tool_parser that does not match the "
                   "model.")
    elif totals.get("undecoded_tool_call"):
        failure = ("the model asked for a tool but the server could not decode "
                   "the request, so no tool ran. Usually a tool_parser that "
                   "does not match the model.")
    elif looks_like_undecoded_tool_call(row.get("answer") or ""):
        # The framework engines' version of the flag above: they keep no
        # trace, so markup left in the answer is the only evidence that the
        # tool never ran. Measured on the AI PC with openai-agents.
        failure = ("the model asked for a tool but the call came back as "
                   "text and was never decoded, so no tool ran.")
    elif getattr(agent, "last_failed", False):
        # A framework adapter that hit its cap or met a model error. Its
        # answer is the failure sentence, which is not empty, so the
        # says_nothing() fallback below scored it ok.
        failure = row["answer"] or "the engine reported a failure."
    elif says_nothing(row.get("answer") or ""):
        # Fallback for the three framework engines, whose traces carry no
        # empty_answer flag: there, an empty answer is the only evidence.
        failure = "the engine returned no answer (reply was empty)."

    if failure is not None:
        row["ok"] = False
        row["error"] = failure
        return row
    row["ok"] = True
    return row


# How long one engine gets before it is called hung. Generous: an 8B model on
# a cold GPU can spend a minute on first token, and a false timeout would look
# exactly like a broken engine.
WORKER_TIMEOUT_S = 600


def _run_isolated(config_path: str, engine: str, question: str,
                  timeout: float = WORKER_TIMEOUT_S) -> dict:
    """Measure one engine in a FRESH PROCESS, and return its row.

    This is what makes Peak MB mean anything. Every engine used to run in one
    process, and RSS is a whole-process number, so each engine inherited
    everything its predecessors had allocated. Measured on the AI PC against
    live OVMS: native read 465.8 MB running first and 1155.6 MB running last,
    against an isolated figure of 466 MB. The lightest engine was reported as
    the heaviest, purely by list position, and reversing the order reversed
    the conclusion. A benchmark that changes its answer when you reorder the
    inputs is not measuring the inputs.

    The child times its OWN build and answer, so interpreter startup and
    import cost land in neither: they are the harness, not the engine.
    """
    import json
    import subprocess
    import sys

    command = [sys.executable, "-m", "ovat.bench", "--worker",
               "--config", config_path, "--engine", engine,
               "--question", question]
    try:
        finished = subprocess.run(command, capture_output=True, text=True,
                                  timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"engine": engine, "ok": False,
                "error": f"TimeoutError: no answer within {timeout:.0f}s",
                "answer": "", "latency_s": None, "build_s": None,
                "peak_rss_mb": None, "prompt_tokens": None,
                "completion_tokens": None, "tool_calls": None}

    # The child prints exactly one JSON object on stdout. Anything else means
    # it died before it could, and its stderr is the only clue we have.
    for line in reversed(finished.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                break
    detail = (finished.stderr or "").strip().splitlines()
    return {"engine": engine, "ok": False,
            "error": f"WorkerError: {detail[-1] if detail else 'no output'}",
            "answer": "", "latency_s": None, "build_s": None,
            "peak_rss_mb": None, "prompt_tokens": None,
            "completion_tokens": None, "tool_calls": None}


def _median(rows: list, key: str):
    """Median of a field over the rows that report it; None if none do.

    Absent stays absent here too: an engine that never reports tokens gets
    None, not a median of zeros.
    """
    import statistics

    values = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.median(values), 3) if values else None


def _aggregate(engine: str, attempts: list) -> dict:
    """Fold N runs of one engine into one row, keeping every attempt.

    `ok` is True only when EVERY run succeeded: an engine that answers 2 runs
    in 3 is the unreliable engine this exists to expose, not a passing one.
    Medians rather than means, so one 150-second outlier (a full KV cache,
    say) does not stand for the typical run.
    """
    from collections import Counter

    good = [a for a in attempts if a["ok"]]
    errors = Counter(a["error"] for a in attempts if not a["ok"])
    return {
        "engine": engine,
        "ok": len(good) == len(attempts),
        "ok_runs": len(good),
        "runs": len(attempts),
        # The commonest failure, which is what the table has room for; every
        # attempt's own error is still in "attempts".
        "error": errors.most_common(1)[0][0] if errors else None,
        "answer": (good or attempts)[-1]["answer"],
        "latency_s": _median(attempts, "latency_s"),
        "build_s": _median(attempts, "build_s"),
        "peak_rss_mb": _median(attempts, "peak_rss_mb"),
        "prompt_tokens": _median(good, "prompt_tokens"),
        "completion_tokens": _median(good, "completion_tokens"),
        "tool_calls": _median(good, "tool_calls"),
        "attempts": attempts,
    }


def benchmark(config, question: str, engines, build_agent=None,
              config_path: str | None = None, repeat: int = 1) -> dict:
    """Run the question through each engine and return a report dict.

    With `config_path`, each engine is measured in its own process, which is
    the only way Peak MB is comparable between rows. Without it the engines
    share this process, which tests want (they inject a fake build_agent) and
    which is why the CLI always passes the path.

    repeat > 1 runs every engine that many times, each in a fresh process
    when isolated, and reports a success RATE. Measured on the AI PC, the
    same prompt at temperature 0.0 called its tool 1 run in 5 on one engine
    and 5 in 5 on another; a single run cannot tell those apart.
    """
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    isolated = config_path is not None and build_agent is None

    def one(engine):
        if isolated:
            return _run_isolated(config_path, engine, question)
        return benchmark_engine(config, engine, question,
                                build_agent=build_agent)

    rows = []
    for engine in engines:
        attempts = [one(engine) for _ in range(repeat)]
        # A single run keeps the row shape it always had.
        rows.append(attempts[0] if repeat == 1
                    else _aggregate(engine, attempts))
    return {
        "model": config.model.name,
        "ovms_url": config.model.ovms_url,
        "question": question,
        "isolated": isolated,
        "repeat": repeat,
        "results": rows,
    }


def _worker_main(argv=None) -> int:
    """`python -m ovat.bench --worker`: measure ONE engine, print one JSON row.

    Deliberately tiny and deliberately quiet: stdout carries the row and
    nothing else, so the parent can parse it without guessing.
    """
    import argparse
    import json

    from ovat.config.workflow import load_workflow

    parser = argparse.ArgumentParser(prog="ovat.bench")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--config", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args(argv)

    try:
        config = load_workflow(args.config)
        row = benchmark_engine(config, args.engine, args.question)
    except Exception as exc:                      # never a traceback on stdout
        row = {"engine": args.engine, "ok": False,
               "error": f"{type(exc).__name__}: {exc}", "answer": "",
               "latency_s": None, "build_s": None, "peak_rss_mb": None,
               "prompt_tokens": None, "completion_tokens": None,
               "tool_calls": None}
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main())
