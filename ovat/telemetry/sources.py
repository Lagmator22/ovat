# ovat/telemetry/sources.py
"""Concrete telemetry sources: agent, process, Intel hardware.

Each fills the TelemetrySource socket. None of them raises: a source that
takes down the run it is measuring has inverted its own purpose.
"""
import json
import os
import re
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
        self._agent = agent

    @property
    def agent(self):
        if callable(self._agent):
            return self._agent()
        return self._agent

    def sample(self) -> dict:
        totals = (getattr(self.agent, "last_trace", None) or {}).get("totals")
        if not totals:
            return {}
        # Absent stays absent. A zero here would read as "used no tokens",
        # which is a different claim from "nobody told us".
        return {k: v for k, v in totals.items() if v is not None}

    @property
    def unavailable(self) -> str | None:
        agent = self.agent
        if agent is None:
            return "no agent has run yet"
        if getattr(agent, "last_trace", None) is None:
            return ("this engine does not record per-turn data; only the "
                    "native loop does")
        return None


class SystemSource(TelemetrySource):
    """CPU, memory and thread counts for this machine and this process.

    This exists because the telemetry page was nearly empty on any machine
    without Intel UT: one metric, one graph. These come from psutil, which
    works on macOS, Windows and Linux alike, so the page is useful everywhere
    and the Intel rows become the EXTRA rather than the only content.

    Per-core percentages are reported individually. An agent that pins one
    core looks identical to an idle machine in an averaged figure, and which
    of those is happening is the whole question when a run feels slow.
    """

    name = "system"

    def __init__(self):
        self._proc = None
        try:
            import psutil
            self._proc = psutil.Process()
            # First call always returns 0.0 and primes the delta; get it out
            # of the way here so the first drawn frame is a real reading.
            psutil.cpu_percent(percpu=True)
            self._proc.cpu_percent()
        except Exception:
            self._proc = None

    def sample(self) -> dict:
        try:
            import psutil
        except ImportError:
            return {}
        out = {}
        try:
            cores = psutil.cpu_percent(percpu=True)
            out["cpu_pct"] = round(sum(cores) / len(cores), 1) if cores else 0.0
            for index, pct in enumerate(cores):
                out[f"cpu{index}_pct"] = pct
        except Exception:
            pass
        try:
            memory = psutil.virtual_memory()
            out["ram_used_pct"] = memory.percent
            out["ram_available_mb"] = round(memory.available / (1024 ** 2), 1)
        except Exception:
            pass
        try:
            freq = psutil.cpu_freq()
            if freq is not None:
                out["cpu_mhz"] = round(freq.current, 0)
        except Exception:
            pass
        if self._proc is not None:
            try:
                out["proc_cpu_pct"] = self._proc.cpu_percent()
                out["threads"] = self._proc.num_threads()
            except Exception:
                pass
        return out

    @property
    def unavailable(self) -> str | None:
        try:
            import psutil     # noqa: F401
        except ImportError:
            return "psutil is not installed"
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
# Folders people actually unzip it into. The release extracts to a VERSIONED
# directory (ut-tool-ext-v0.2.0-beta1.1), so every entry here is also searched
# one level deep for anything starting "ut-" or "ut_". Without that, unzipping
# into ~/ut leaves the binary at ~/ut/ut-tool-ext-v0.2.0-beta1.1/bin/ut.exe,
# which a flat check misses entirely.
_UT_KNOWN_DIRS = [
    os.path.expanduser(p) for p in (
        [r"~\ut", r"~\ut-tool", r"~\Downloads", r"C:\ut", r"C:\ut-tool",
         r"C:\Program Files\Intel\ut"]
        if sys.platform == "win32" else
        ["~/ut", "~/ut-tool", "~/Downloads", "/opt/intel/ut", "/opt/ut"]
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
        # One level deeper, for the versioned folder the zip extracts to.
        try:
            entries = sorted(os.listdir(folder), reverse=True)
        except OSError:
            continue
        for entry in entries:
            if not entry.lower().startswith(("ut-", "ut_")):
                continue
            found = as_binary(os.path.join(folder, entry))
            if found:
                # reverse-sorted above, so the newest version wins when
                # several releases are unzipped side by side.
                return found, f"known location ({folder}/{entry})"
    return None, ("not found. Unzip the ut-tool release and either set "
                  "OVAT_UT to that folder or put it in ~/ut")


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

    #: What UT's collectors report, and therefore what the page can draw.
    #: level-zero gives per-engine device activity on GPU and NPU; socwatch
    #: gives package power and frequency. Names are normalised to snake_case
    #: here so the page does not have to know UT's exact spelling, which has
    #: differed between beta builds.
    KNOWN_METRICS = (
        "npu_utilization", "gpu_utilization", "gpu_render_pct",
        "gpu_media_pct", "gpu_compute_pct", "gpu_copy_pct",
        "power_w", "package_power_w", "gpu_power_w", "npu_power_w",
        "gpu_freq_mhz", "cpu_freq_mhz", "gpu_mem_used_mb",
        "gpu_mem_bandwidth_gbs", "temperature_c",
    )

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
        return self._normalise(frame)

    @staticmethod
    def _normalise(frame: dict) -> dict:
        """Flatten and snake_case whatever UT emitted.

        UT's JSON has varied between beta builds (nested per-collector
        objects in some, flat in others) and uses mixed casing. Normalising
        here means the page and the tests do not have to track that, and a
        build that renames a field degrades to a missing metric rather than a
        crash.
        """
        flat = {}

        def walk(obj, prefix=""):
            for key, value in (obj or {}).items():
                # Anything that is not a word character becomes an
                # underscore: UT writes "Render %" and "GPU-Freq(MHz)", and a
                # metric name is used as a CSS id on the page, where those
                # would be invalid.
                name = re.sub(r"[^0-9A-Za-z]+", "_", f"{prefix}{key}").strip("_")
                if isinstance(value, dict):
                    walk(value, f"{name}_")
                elif isinstance(value, (int, float)) and not isinstance(
                        value, bool):
                    flat[name.lower()] = value

        walk(frame)
        return flat

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
