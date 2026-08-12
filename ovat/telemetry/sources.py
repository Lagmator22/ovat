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
            # Say the CONDITION, not the symptom. "no agent has run yet" reads
            # as a bug when a chat is plainly answering two screens away; the
            # truth is that only /engine ovms builds an agent at all, and only
            # once a question has been answered does it have totals to report.
            return ("open /chat, switch to /engine ovms and ask something; "
                    "the local engine is retrieval, not an agent loop")
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
        # A DIRECTORY, and one that ut is actually told about. The old code
        # made a temp FILE and never passed it on the command line, so ut
        # ignored it entirely and dumped its own binary traces
        # (ut_default_output.*.bin, tens of MB) into whatever directory the
        # user happened to run from. stop() then deleted the untouched temp
        # file and left the real output behind -- so every `ovat run
        # --telemetry` quietly added to a pile in the user's project folder.
        self._out_path = tempfile.mkdtemp(prefix="ovat-ut-")
        try:
            self._proc = subprocess.Popen(
                [self.binary, "--continuous",
                 "--enable", self.collectors,
                 "--sampling-interval", str(self.sampling_interval_ms),
                 "--output", self._out_path],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            # A profiler that cannot start must not stop the agent running.
            self._proc = None

    def sample(self) -> dict:
        """One snapshot of whatever UT has emitted since the last read.

        MEASURED ON THE AI PC, and not what the README implied: continuous
        mode does not stream JSON on stdout. It writes binary traces
        (ut_default_output.l0_gpu.bin, .l0_npu.bin, .ut.main.bin) which need
        bin2perfetto to decode. So the process starts, reports "live", and
        emits nothing this can read.

        Both shapes are handled: a JSON line if a build ever streams one, and
        "name: value" or "name = value" text otherwise. Anything unparseable
        is skipped rather than raised, because a malformed frame is a missing
        reading and not a reason to lose the run.
        """
        if self._proc is None or self._proc.stdout is None:
            return {}
        line = self._proc.stdout.readline()
        if not line:
            return {}
        line = line.strip()
        if line and not line.startswith("{"):
            return self._parse_text_line(line)
        try:
            frame = json.loads(line)
        except ValueError:
            return {}
        return self._normalise(frame)

    @staticmethod
    def _parse_text_line(line: str) -> dict:
        """A "name: value" or "name = value" line into one metric.

        UT's human-readable output is the fallback when a build does not
        stream JSON. Lines that carry no number are not measurements and are
        dropped rather than guessed at.
        """
        for separator in (":", "="):
            name, found, value = line.partition(separator)
            if not found:
                continue
            try:
                number = float(value.strip().split()[0])
            except (ValueError, IndexError):
                return {}
            return IntelHardwareSource._normalise({name.strip(): number})
        return {}

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

    @property
    def note(self) -> str | None:
        """Extra context the page shows beside a live source.

        "live" with no metrics behind it is the most confusing state a source
        can be in: it looks like an idle NPU rather than an unreadable one.
        """
        if self.unavailable is None and self._proc is not None:
            return ("running, but this UT build writes binary traces rather "
                    "than streaming numbers; decode with bin2perfetto")
        return None

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
                # rmtree, not remove: it is a directory now, and ut writes its
                # binary traces inside it. Removing only the directory entry
                # would leave those behind, which is the leak this fixes.
                shutil.rmtree(self._out_path)
            except OSError:
                pass
            self._out_path = None


#: Where the Linux NPU driver publishes its busy-time counter. The kernel
#: module is `intel_vpu` (drivers/accel/ivpu), and it exposes one directory per
#: PCI device, each carrying `npu_busy_time_us`.
_NPU_SYSFS_ROOT = "/sys/bus/pci/drivers/intel_vpu"


class NPUSource(TelemetrySource):
    """NPU utilisation, from the driver rather than from a profiler.

    WHY THIS EXISTS ALONGSIDE IntelHardwareSource. Intel UT is the rich
    answer -- power, per-engine timelines, thermals -- but its continuous mode
    writes binary traces that need bin2perfetto to decode, so it produces no
    number this page can draw. A live utilisation percentage is the number
    actually being asked for, and on Linux the driver publishes it directly.

    THE MEASUREMENT. `npu_busy_time_us` is a monotonic counter of microseconds
    the NPU spent executing jobs. A single reading is meaningless; two readings
    give a duty cycle:

        busy% = delta_busy_us / (delta_wall_s * 10_000)

    which is the same arithmetic btop uses for its Intel NPU meter. The first
    sample after start() therefore reports nothing -- there is no previous
    reading to subtract -- rather than reporting 0, which would draw an idle
    NPU during the one second an agent is most likely to be using it.

    PLATFORM HONESTY. Windows exposes NPU utilisation to Task Manager through
    DXCore, not through a documented performance-counter path, so there is no
    verified way to read it from here yet and this source says so instead of
    guessing. macOS has no Intel NPU at all.
    """

    name = "npu"

    def __init__(self, sysfs_root: str = _NPU_SYSFS_ROOT):
        # Injectable so the reader can be tested against a fake sysfs tree on
        # any OS. A test that needs real NPU hardware is a test that never
        # runs.
        self.sysfs_root = sysfs_root
        self._previous: tuple[float, int] | None = None

    def _device_dirs(self) -> list[str]:
        """Every intel_vpu device directory that publishes a busy counter."""
        try:
            entries = sorted(os.listdir(self.sysfs_root))
        except OSError:
            return []
        found = []
        for entry in entries:
            path = os.path.join(self.sysfs_root, entry, "npu_busy_time_us")
            if os.path.exists(path):
                found.append(os.path.join(self.sysfs_root, entry))
        return found

    @property
    def unavailable(self) -> str | None:
        if sys.platform == "darwin":
            return "no Intel NPU on macOS"
        if sys.platform.startswith("win"):
            return ("Windows exposes NPU usage through DXCore, not a "
                    "documented perf counter. Check whether this build has "
                    "one: Get-Counter -ListSet *NPU*")
        if not self._device_dirs():
            return (f"no NPU found under {self.sysfs_root}. The intel_vpu "
                    f"driver provides it; check `lsmod | grep intel_vpu`")
        return None

    def sample(self) -> dict:
        devices = self._device_dirs()
        if not devices:
            return {}
        # One NPU per machine in practice; the first device is the NPU.
        device = devices[0]
        now = time.monotonic()
        try:
            with open(os.path.join(device, "npu_busy_time_us")) as handle:
                busy_us = int(handle.read().strip())
        except (OSError, ValueError):
            return {}

        out = {}
        if self._previous is not None:
            previous_time, previous_busy = self._previous
            elapsed = now - previous_time
            # A counter that went backwards means the driver reloaded; treat
            # it as a fresh start rather than reporting a negative percentage.
            if elapsed > 0 and busy_us >= previous_busy:
                percent = (busy_us - previous_busy) / (elapsed * 10_000)
                out["utilization"] = round(min(100.0, max(0.0, percent)), 1)
        self._previous = (now, busy_us)

        # Optional on some driver versions, so its absence is not an error.
        try:
            with open(os.path.join(device, "npu_memory_utilization")) as handle:
                out["memory_mb"] = round(int(handle.read().strip()) / 1024, 1)
        except (OSError, ValueError):
            pass
        return out


#: OVMS's own log line for KV cache state, from its continuous-batching
#: executor (src/llm/language_model/continuous_batching/llm_executor.hpp):
#:
#:     type: dynamic, cache usage: 98.5% of 3.60 GB
#:
#: Parsed rather than invented. The percentage is the number that matters; the
#: size is carried alongside so "98% of 1 GB" and "98% of 16 GB" are not
#: reported as the same situation.
_CACHE_USAGE = re.compile(
    r"cache usage:\s*([\d.]+)\s*%\s*of\s*([\d.]+)\s*([KMGT]?B)", re.IGNORECASE)

_UNIT_GB = {"B": 1 / (1024 ** 3), "KB": 1 / (1024 ** 2), "MB": 1 / 1024,
            "GB": 1.0, "TB": 1024.0}


class OVMSLogSource(TelemetrySource):
    """KV cache usage, read out of the OVMS log.

    WHY A LOG AND NOT AN ENDPOINT. OVMS has a Prometheus `/metrics` endpoint,
    but its own documentation states that "metrics related to text generation
    are not exposed via metrics endpoint" -- KV cache usage is written to the
    server log and nowhere else. So a log reader is not a shortcut here; it is
    the only interface that carries this number.

    WHY IT MATTERS. Tool calls stopped decoding on a long-lived server whose
    cache was reported at 98-100%, and decoded cleanly on a fresh one. That
    correlation is the whole basis of the KV-cache explanation, and until now
    the figure behind it had to be read by hand out of a log file at the exact
    moment of failure. Exposing it as a source means `ovat run --telemetry`
    captures the cache state and the failing trace in the same recording,
    which is what the explanation needs in order to stop being a hypothesis.

    Only the LAST reading in the file is reported: the log accumulates one
    line per scheduler tick, and the current state is the one being asked for.
    """

    name = "ovms"

    #: Reading the whole log every tick would be O(file) forever, and this file
    #: grows for as long as the server runs. The last lines are the current
    #: state, so only the tail is read.
    TAIL_BYTES = 65_536

    def __init__(self, log_path: str = "ovms.log"):
        self.log_path = log_path

    @property
    def unavailable(self) -> str | None:
        if not os.path.exists(self.log_path):
            return (f"no OVMS log at {self.log_path}. This source reads the "
                    f"log `ovat serve` writes; start a server first")
        return None

    @property
    def note(self) -> str | None:
        """Live, but the log carries no cache line yet.

        A server that has started but not yet served a generation request has
        written no cache line at all. That is a different state from "no
        server", and reporting it as an absent source would be wrong.
        """
        if self.unavailable is None and not self._last_reading():
            return ("OVMS is running but has logged no KV cache line yet; it "
                    "appears once a generation request has been served")
        return None

    def _last_reading(self) -> tuple[float, float] | None:
        """(percent, gigabytes) from the most recent cache line, or None."""
        try:
            size = os.path.getsize(self.log_path)
            with open(self.log_path, "r", encoding="utf-8",
                      errors="replace") as handle:
                if size > self.TAIL_BYTES:
                    handle.seek(size - self.TAIL_BYTES)
                    handle.readline()          # discard a partial first line
                text = handle.read()
        except OSError:
            return None

        found = _CACHE_USAGE.findall(text)
        if not found:
            return None
        percent, amount, unit = found[-1]
        try:
            gigabytes = float(amount) * _UNIT_GB.get(unit.upper(), 1.0)
            return float(percent), round(gigabytes, 2)
        except ValueError:
            return None

    def sample(self) -> dict:
        reading = self._last_reading()
        if reading is None:
            return {}
        percent, gigabytes = reading
        return {"kv_cache_pct": percent, "kv_cache_gb": gigabytes}
