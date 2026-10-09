# ovat/agent/usage.py
"""The run trace's totals for the three framework engines.

The native loop builds its own trace turn by turn (loop.py). The framework
engines own their request loops, so each adapter reads the usage its
framework kept and hands it here. One function builds the totals, so bench,
`ovat run --trace` and the telemetry source read the same shape from all
four engines with no special case for any of them.
"""
import time


def sum_known(values):
    """Sum the values that were reported, or None if none was.

    Absent is not zero. A server that sends no usage block would otherwise
    produce "prompt_tokens": 0, which reads as "this run used no tokens"
    rather than "nobody told us".
    """
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def framework_trace(engine: str, calls: list | None, tool_calls: int | None,
                    failed: bool, started: float) -> dict:
    """A last_trace in the native loop's shape, from what a framework kept.

    `calls` holds one (prompt_tokens, completion_tokens) pair per model call,
    with None where that reply carried no usage, or is None when the
    framework handed back nothing at all (LangChain's step cap). `tool_calls`
    counts tools that actually ran, or is None when the framework did not say.
    """
    if calls is None:
        calls, turns = [], None
    else:
        turns = len(calls)
    return {"engine": engine, "totals": {
        "turns": turns,
        "latency_s": round(time.monotonic() - started, 3),
        "prompt_tokens": sum_known(p for p, _ in calls),
        "completion_tokens": sum_known(c for _, c in calls),
        "tool_calls": tool_calls,
        "failed": failed,
    }}
