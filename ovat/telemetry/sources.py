# ovat/telemetry/sources.py
"""Concrete telemetry sources: agent, process, Intel hardware.

Each fills the TelemetrySource socket. None of them raises: a source that
takes down the run it is measuring has inverted its own purpose.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from ovat.telemetry.base import TelemetrySource


class AgentTraceSource(TelemetrySource):
    """The numbers the native loop already records, as a source.

    `loop.py` has filled `last_trace` since W9; this exposes it through the
    contract so the JSON file, an OTLP exporter and the TUI page all read one
    thing instead of three.
    """

    name = "agent"

    def __init__(self, agent=None):
        self.agent = agent

    def sample(self) -> dict:
        totals = (getattr(self.agent, "last_trace", None) or {}).get("totals")
        if not totals:
            return {}
        # Absent stays absent. A zero here would read as "used no tokens",
        # which is a different claim from "nobody told us".
        return {k: v for k, v in totals.items() if v is not None}

    @property
    def unavailable(self) -> str | None:
        if self.agent is None:
            return "no agent has run yet"
        if getattr(self.agent, "last_trace", None) is None:
            return ("this engine does not record per-turn data; only the "
                    "native loop does")
        return None


class ProcessMemorySource(TelemetrySource):
    """This process's resident memory, sampled live.

    The proposal's <8 GB criterion is about the toolkit's own footprint, and
    the model's several gigabytes live in OVMS, not here. Says so rather than
    letting the number be read as the whole system's.
    """

    name = "process"

    def sample(self) -> dict:
        try:
            import psutil
        except ImportError:
            return {}
        try:
            rss = psutil.Process().memory_info().rss
        except Exception:
            return {}
        return {"rss_mb": round(rss / (1024 * 1024), 1)}

    @property
    def unavailable(self) -> str | None:
        try:
            import psutil     # noqa: F401
        except ImportError:
            return "psutil is not installed"
        return None


# Where Intel's Unified Telemetry tool is unzipped. Same problem as OVMS: it
# ships as a zip, is set up with ut-vars.cmd, and is essentially never on PATH.
_UT_EXE = "ut.exe" if sys.platform == "win32" else "ut"
_UT_KNOWN_DIRS = [
    os.path.expanduser(p) for p in (
        [r"~\ut-tool", r"~\Downloads\ut-tool-ext", r"C:\ut-tool"]
        if sys.platform == "win32" else
        ["~/ut-tool", "~/Downloads/ut-tool-ext", "/opt/intel/ut"]
    )
]


def find_ut(explicit: str | None = None) -> tuple:
    """Locate Intel UT. Returns (path or None, how it was found).

    Deliberately the same shape as ovat/core/ovms_locator.find_ovms, because
    it is the same problem: an Intel tool that arrives as a zip and is never
    on PATH. Order: explicit config, OVAT_UT env, PATH, known unzip folders.
    """
    def as_binary(path: str) -> str | None:
        path = os.path.expanduser(path)
        if os.path.isfile(path):
            return path
        for candidate in (os.path.join(path, _UT_EXE),
                          os.path.join(path, "bin", _UT_EXE)):
            if os.path.isfile(candidate):
                return candidate
        return None

    if explicit:
        found = as_binary(explicit)
        return (found, "config") if found else (
            None, f"{explicit!r} has no {_UT_EXE} in it")

    env = os.environ.get("OVAT_UT")
    if env:
        found = as_binary(env)
        return (found, "OVAT_UT env") if found else (
            None, f"OVAT_UT points at {env!r} but no {_UT_EXE} is there")

    on_path = shutil.which("ut")
    if on_path:
        return on_path, "PATH"

    for folder in _UT_KNOWN_DIRS:
        found = as_binary(folder)
        if found:
            return found, f"known location ({folder})"
    return None, "not found (config, OVAT_UT, PATH, known folders all empty)"


class IntelHardwareSource(TelemetrySource):
    """GPU/NPU utilisation and power, via Intel's Unified Telemetry tool.

    This is the Intel-native half, and the number the proposal's stretch goal
    actually wants: NPU power draw beside agent token counts. Nothing else in
    the OpenVINO ecosystem puts those two on one screen.

    UT is a SUBPROCESS PROFILER, not a library. There is no Python API: you
    run `ut -c` for continuous sampling or `ut -a "<cmd>"` to wrap a command.
    So this shells out and parses, exactly as ModelManager does for ovms.

    Platform truth: the tool ships as .exe/.dll for Windows, with a Linux
    build. It does NOT run on macOS, so on a Mac this reports unavailable
    with a sentence rather than sampling zeros, which would be
    indistinguishable from an idle NPU.
    """

    name = "intel"

    def __init__(self, ut_binary: str | None = None,
                 sampling_interval_ms: int = 500,
                 collectors: str = "socwatch,level-zero"):
        self.binary, self.how = find_ut(ut_binary)
        self.sampling_interval_ms = sampling_interval_ms
        # socwatch = power, level-zero = GPU/NPU engine activity. emon (CPU
        # perf counters) is left off by default: it needs a kernel driver and
        # is far heavier than a background sampler should be.
        self.collectors = collectors
        self._proc = None
        self._out_path = None

    @property
    def unavailable(self) -> str | None:
        if sys.platform == "darwin":
            return ("Intel Unified Telemetry does not run on macOS; hardware "
                    "telemetry needs the AI PC")
        if self.binary is None:
            return (f"Intel UT not found: {self.how}. Unzip the ut-tool "
                    f"release and set OVAT_UT to that folder.")
        return None

    def start(self) -> None:
        """Begin continuous collection in the background."""
        if self.unavailable or self._proc is not None:
            return
        handle, self._out_path = tempfile.mkstemp(suffix=".json",
                                                  prefix="ovat-ut-")
        os.close(handle)
        try:
            self._proc = subprocess.Popen(
                [self.binary, "--continuous",
                 "--enable", self.collectors,
                 "--sampling-interval", str(self.sampling_interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            # A profiler that cannot start must not stop the agent running.
            self._proc = None

    def sample(self) -> dict:
        """One snapshot of whatever UT has emitted since the last read.

        Continuous mode streams JSON lines on stdout. Anything unparseable is
        skipped rather than raised: a malformed frame is a missing reading,
        not a reason to lose the run.
        """
        if self._proc is None or self._proc.stdout is None:
            return {}
        line = self._proc.stdout.readline()
        if not line:
            return {}
        try:
            frame = json.loads(line)
        except ValueError:
            return {}
        return {k: v for k, v in frame.items() if isinstance(v, (int, float))}

    def stop(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None
        if self._out_path and os.path.exists(self._out_path):
            try:
                os.remove(self._out_path)
            except OSError:
                pass
            self._out_path = None
