# ovat/bench.py
"""Run one question through several engines and measure what each one costs.

This is the proposal's AI PC profiling deliverable. The claim OVAT makes is
"one YAML, any framework, the same OVMS backend", and the only way to show
that is worth anything is to run the SAME question through every engine
against the SAME server and put the numbers side by side.

Kept out of cli/main.py because it is real logic with real edge cases and
deserves its own tests, not a place inside a typer command body.

Two measurement honesty notes, both of which shape the output:

  * Token counts come from OVMS's usage field, which only the NATIVE loop
    records (it owns its own request loop; the frameworks own theirs and do
    not hand the usage back). So tokens are reported where they are known and
    left explicitly unknown elsewhere, never zero. A zero would read as "this
    engine used no tokens", which is false.
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
        row["latency_s"] = round(time.monotonic() - run_started, 3)

    row["peak_rss_mb"] = memory.peak_mb
    row["ok"] = True

    # Tokens only where the engine actually recorded them. See the module
    # docstring: absent is not zero.
    totals = (getattr(agent, "last_trace", None) or {}).get("totals") or {}
    for key in ("prompt_tokens", "completion_tokens", "tool_calls"):
        if totals.get(key) is not None:
            row[key] = totals[key]
    return row


def benchmark(config, question: str, engines, build_agent=None) -> dict:
    """Run the question through each engine in turn. Returns a report dict."""
    rows = [benchmark_engine(config, engine, question, build_agent=build_agent)
            for engine in engines]
    return {
        "model": config.model.name,
        "ovms_url": config.model.ovms_url,
        "question": question,
        "results": rows,
    }
