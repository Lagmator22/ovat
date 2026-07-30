# ovat/telemetry/sinks.py
"""Concrete telemetry sinks: a JSON file, and a live in-memory buffer.

Each fills the TelemetrySink socket. None of them raises: a failing exporter
must not break the run it is measuring, which is the whole reason record() is
specified that way.
"""
import json
import time
from collections import deque

from ovat.telemetry.base import TelemetrySink


class JSONFileSink(TelemetrySink):
    """Append snapshots to a file, one JSON object per line.

    JSON Lines rather than one big array on purpose: a run that is killed
    half-way still leaves a readable file, and `tail -f` works while it runs.
    A single array would need the closing bracket that a crashed run never
    writes.
    """

    name = "jsonl"

    def __init__(self, path: str):
        self.path = path
        self._handle = None

    def record(self, sample: dict) -> None:
        if not sample:
            return
        try:
            if self._handle is None:
                self._handle = open(self.path, "a", encoding="utf-8")
            self._handle.write(json.dumps(sample) + "\n")
            self._handle.flush()
        except Exception:
            # A full disk must not end the agent run. The measurement is lost;
            # the work is not.
            pass

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None


class LiveBufferSink(TelemetrySink):
    """The last N snapshots, in memory, for the TUI page to draw.

    A ring buffer rather than a growing list: a telemetry page left open for
    an hour must not be the reason the process runs out of memory. `maxlen`
    on a deque drops the oldest for free.

    `series()` is what a Sparkline wants: one metric's history as a flat list
    of numbers, oldest first.
    """

    name = "live"

    def __init__(self, capacity: int = 120):
        # 120 at the default 1s tick is two minutes of history, which is
        # about as much as fits a terminal-width sparkline meaningfully.
        self.samples: deque = deque(maxlen=capacity)

    def record(self, sample: dict) -> None:
        if not sample:
            return
        stamped = dict(sample)
        stamped.setdefault("t", time.time())
        self.samples.append(stamped)

    def series(self, metric: str) -> list:
        """One metric's history, oldest first, skipping frames that lack it.

        Skipping rather than zero-filling: a gap in a source is not a reading
        of zero, and a sparkline that dips to the floor whenever a collector
        stalls is a graph that lies.
        """
        return [s[metric] for s in self.samples
                if isinstance(s.get(metric), (int, float))]

    def latest(self, metric: str, default=None):
        """The most recent value of one metric, or `default` if never seen."""
        for sample in reversed(self.samples):
            value = sample.get(metric)
            if isinstance(value, (int, float)):
                return value
        return default

    def metrics(self) -> list:
        """Every metric name seen so far, in first-seen order.

        The page does not know in advance which sources are live, so it asks
        rather than hardcoding a list that would go stale the moment a source
        is added.
        """
        seen = {}
        for sample in self.samples:
            for key in sample:
                if key != "t" and isinstance(sample[key], (int, float)):
                    seen[key] = True
        return list(seen)

    def close(self) -> None:
        self.samples.clear()


class FanOutSink(TelemetrySink):
    """Deliver each snapshot to several sinks.

    Lets `ovat run --telemetry out.jsonl` and the live TUI page consume the
    same stream without either knowing about the other. One failing sink does
    not stop the rest, for the same reason record() never raises.
    """

    name = "fanout"

    def __init__(self, *sinks):
        self.sinks = list(sinks)

    def record(self, sample: dict) -> None:
        for sink in self.sinks:
            try:
                sink.record(sample)
            except Exception:
                pass

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                pass
