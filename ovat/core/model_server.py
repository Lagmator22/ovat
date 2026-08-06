# ovat/core/model_server.py
"""Layer 8: manages the OVMS process lifecycle: start, health-check, stop.

It's "process lifecycle management", which kinda sounds fancy but is just
subprocess.Popen + polling a health URL. Key proposal point: switching models
= stop() then start() (a fresh restart), NOT an in-process swap; that's the
mitigation for the memory-leak risk even though in future we can get it to 
continue even with a new model.

Uses urllib (standard library) for health checks, so there's no extra
dependency. Needs the `ovms` binary to actually start (Linux/Windows/Docker).
"""
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request

# Where `ovat serve` records the OVMS process id. A pidfile is the classic
# daemon pattern: the starting process exits, but the NUMBER of the server
# process survives on disk, so a later `ovat serve --stop` can find and stop
# a server it did not start itself.
DEFAULT_PID_PATH = "ovms.pid"


class ModelServer:
    """Manages the OVMS process: start, wait-until-ready, stop."""

    def __init__(self, model_name: str, source_model: str | None = None,
                 model_repository_path: str = "models", device: str = "CPU",
                 port: int = 8000, tool_parser: str = "hermes3",
                 reasoning_parser: str | None = None,
                 task: str = "text_generation",
                 enable_prefix_caching: bool = True,
                 binary: str = "ovms"):
        self.model_name = model_name
        # Note to myself: source_model is the Hugging Face id OVMS downloads if
        # the model is not already on disk, for example OpenVINO/Qwen3-8B-int4-ov.
        self.source_model = source_model
        # The folder on disk where models live. OVMS needs this to have anything
        # to serve. Leaving it out is exactly why the server used to exit at once.
        self.model_repository_path = model_repository_path
        self.device = device
        self.port = port
        self.tool_parser = tool_parser
        self.reasoning_parser = reasoning_parser
        # task text_generation is what turns on the chat endpoints I call.
        self.task = task
        self.enable_prefix_caching = enable_prefix_caching
        # The resolved executable ("ovms" if on PATH, or a full path found by
        # ovms_locator). Launching by full path means no setupvars.bat needed.
        self.binary = binary
        self.process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v3"

    @property
    def health_url(self) -> str:
        return f"http://localhost:{self.port}/v2/health/ready"

    def start(self, log_path: str = "ovms.log",
              pid_path: str = DEFAULT_PID_PATH) -> None:
        """Launch OVMS in the background. (Flags illustrative -> match them to
        our OVMS version / the demo README.)"""
        # Note to myself: the three flags that matter most here are
        # model_repository_path, source_model and task. Without a model to load,
        # OVMS starts and exits in under a second because it has nothing to do.
        cmd = [
            self.binary,
            "--rest_port", str(self.port),
            "--model_repository_path", self.model_repository_path,
            "--model_name", self.model_name,
            "--task", self.task,
            "--target_device", self.device,
        ]
        # "auto" means "say nothing and let OVMS decide". OVMS inspects the
        # model's chat template at startup and picks the parser itself, but an
        # explicit --tool_parser always wins, so passing one unconditionally
        # made auto-detection unreachable. That matters for any family newer
        # than OVAT's own list: Qwen3.5 emits qwen3coder-shaped tool calls
        # (<function=..><parameter=..>), not hermes3 JSON, and OVMS ships no
        # qwen3_5 parser -- forcing hermes3 there decodes nothing at all.
        if self.tool_parser and self.tool_parser != "auto":
            cmd += ["--tool_parser", self.tool_parser]
        # A knob, not a constant: some OVMS builds/devices reject the flag,
        # and turning it off should be a YAML edit, not a code edit.
        if self.enable_prefix_caching:
            cmd += ["--enable_prefix_caching", "true"]
        if self.reasoning_parser:
            cmd += ["--reasoning_parser", self.reasoning_parser]
        # I only add source_model when I have one. On the first run OVMS uses it
        # to download the model, about 5 GB, then later runs see it on disk.
        if self.source_model:
            cmd += ["--source_model", self.source_model]
        # Note: send OVMS logs to a FILE, not subprocess.PIPE. Piping without
        # ever draining the pipe deadlocks OVMS once its output fills the ~64KB
        # OS pipe buffer. A file has no such limit, and it lets me show the real
        # error if OVMS fails to start.
        self.log_path = log_path
        self._log_file = open(log_path, "w", encoding="utf-8")
        try:
            # Give the child what setupvars.bat would have given it, so no
            # user has to source anything by hand.
            #
            # The binary's own folder is not enough. The python_on build --
            # which the README tells users to download, because python_off
            # cannot do tool calling -- links python3xx.dll out of
            # <ovms>/python, NOT the folder next to ovms.exe. Windows' "search
            # the exe's own directory first" rule therefore never finds it,
            # and the process dies instantly with 0xC0000135 DLL_NOT_FOUND,
            # before it can write one byte to the log. `ovat serve` reported
            # "OVMS exited without becoming ready" over an empty file.
            #
            # setupvars.bat sets PYTHONHOME and adds python/ and
            # python/Scripts; all three are needed and all three are done here.
            # Only when that folder EXISTS: a python_off install has no
            # python/ directory, and pointing PYTHONHOME at a missing path
            # would break the build that currently works.
            env = dict(os.environ)
            binary_dir = os.path.dirname(os.path.abspath(self.binary))
            if os.path.isdir(binary_dir):
                extra = [binary_dir]
                python_home = os.path.join(binary_dir, "python")
                if os.path.isdir(python_home):
                    extra += [python_home, os.path.join(python_home, "Scripts")]
                    env["PYTHONHOME"] = python_home
                env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
            # Windows console semantics: a child launched from a console is
            # ATTACHED to it. Closing that window sends CTRL_CLOSE_EVENT to every
            # attached process, and a Ctrl-C typed at the parent goes to the whole
            # process group. serve is meant to hand the prompt back and leave OVMS
            # running, so both of those would take the server down with the shell.
            # DETACHED_PROCESS gives OVMS no console to be attached to;
            # CREATE_NEW_PROCESS_GROUP puts it outside the parent's Ctrl-C group.
            # (This does NOT cover a parent inside a job object with
            # KILL_ON_JOB_CLOSE; that would need CREATE_BREAKAWAY_FROM_JOB.)
            # POSIX has no equivalent and subprocess REJECTS a non-zero value
            # there with ValueError, so off Windows it stays exactly 0.
            creationflags = 0
            if os.name == "nt":
                creationflags = (subprocess.DETACHED_PROCESS
                                 | subprocess.CREATE_NEW_PROCESS_GROUP)
            self.process = subprocess.Popen(
                cmd, stdout=self._log_file, stderr=subprocess.STDOUT, env=env,
                creationflags=creationflags,
            )
            # Record the pid so `ovat serve --stop` (a NEW process, long after this
            # one exited) can still find and stop the server.
            self.pid_path = pid_path
            with open(pid_path, "w", encoding="utf-8") as f:
                f.write(str(self.process.pid))
        except Exception:
            self.stop()
            raise

    # How long NOTHING may happen before we call it stalled. This is a stall
    # budget, not a deadline: it restarts every time the server makes visible
    # progress, so a download that keeps moving is never interrupted.
    #
    # There is deliberately no absolute cap. A first run has to fetch the
    # model, and that time is unbounded -- gigabytes over whatever link the
    # user has, which can be all night. The old code used a fixed 120s and so
    # could not survive even a fast first run (measured: ~185s for Qwen3.5-4B
    # on the AI PC). Raising the number does not fix the shape of the problem:
    # any fixed value is either too small for a slow link or too large to
    # notice a real hang. Watching progress instead is what curl does with
    # --speed-time/--speed-limit and wget with --read-timeout.
    DEFAULT_STALL_TIMEOUT = 300

    def _progress_marker(self) -> tuple:
        """A cheap fingerprint of "is anything still happening".

        The log file grows as OVMS reports what it is doing, and the model
        repository grows as bytes land on disk. Either moving means the server
        is alive and working, even when the health endpoint is still refusing
        connections. Both are wrapped in try/except because the log may not
        exist yet and the repository may not exist at all on a first run.
        """
        log_size = 0
        try:
            log_size = os.path.getsize(self.log_path)
        except OSError:
            pass
        repo_size = 0
        try:
            for root, _dirs, files in os.walk(self.model_repository_path):
                for name in files:
                    try:
                        repo_size += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        pass
        except OSError:
            pass
        return (log_size, repo_size)

    def wait_until_ready(self, stall_timeout: float = DEFAULT_STALL_TIMEOUT,
                         on_progress=None, max_polls: int | None = None) -> bool:
        """Wait for OVMS to answer its health endpoint.

        Returns False only when the server EXITS, or when it makes no visible
        progress for `stall_timeout` seconds. A download that keeps advancing
        is waited on for as long as it takes.

        on_progress(elapsed_seconds) fires on every poll, so a caller can show
        that something is still happening; silence followed by a failure reads
        as a hang even when it was a download working normally.

        max_polls bounds the loop for tests, which drive a fake clock.
        """
        start = time.time()
        last_change = start
        marker = self._progress_marker()
        polls = 0
        while True:
            # If OVMS already exited, stop waiting. The reason is in the log
            # file (self.log_path) instead of being hidden.
            if self.process is not None and self.process.poll() is not None:
                return False
            try:
                # An opener with an EMPTY ProxyHandler, not urlopen. urlopen
                # honours getproxies(), which reads HTTP_PROXY / ALL_PROXY from
                # the environment -- so on a work laptop behind a proxy or VPN
                # the check for http://localhost:8000 is routed out to that
                # proxy, which refuses it. OVMS would be serving perfectly and
                # `ovat serve` would report that it never came up.
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({}))
                with opener.open(self.health_url, timeout=2) as r:
                    if r.status == 200:
                        return True
            except (urllib.error.URLError, OSError):
                pass  # not up yet, keep waiting

            current = self._progress_marker()
            if current != marker:
                marker = current
                last_change = time.time()      # still working: reset the clock
            elif time.time() - last_change >= stall_timeout:
                return False                   # nothing has moved: really stuck

            if on_progress is not None:
                on_progress(time.time() - start)
            polls += 1
            if max_polls is not None and polls >= max_polls:
                return False
            time.sleep(2)

    def stop(self) -> None:
        """Stop OVMS: ask politely (SIGTERM), then force (SIGKILL).

        terminate() is only a REQUEST the process may ignore; kill() cannot be
        ignored. The finally makes sure the log file handle is released even if
        the process fights the whole way down; before this, a TimeoutExpired
        skipped the close() and leaked the handle.
        """
        try:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()      # escalation: no more asking
                    self.process.wait()
                self.process = None
        finally:
            if getattr(self, "_log_file", None):
                self._log_file.close()
                self._log_file = None
            pid_path = getattr(self, "pid_path", None)
            if pid_path and os.path.exists(pid_path):
                os.remove(pid_path)


def _pid_is_running(pid: int) -> bool:
    """Is this pid alive? Asked WITHOUT signalling anything.

    The obvious implementation is `os.kill(pid, 0)`, and on POSIX that is
    exactly right: signal 0 is the classic existence check and delivers
    nothing. On Windows it is a console grenade.

    Windows has no signals. CPython implements os.kill by mapping sig 0 and 1
    onto GenerateConsoleCtrlEvent, because signal.CTRL_C_EVENT == 0 and
    CTRL_BREAK_EVENT == 1. That call is not addressed to `pid` at all: it
    delivers Ctrl-C to EVERY process sharing the console. The caller, the
    shell it was typed into, and anything running above them.

    Measured on the AI PC: a single os.kill(pid, 0) killed both the probing
    process and its child with KeyboardInterrupt. Since the caller below
    polls this every 0.2s for wait_seconds, `ovat serve --stop` was firing
    roughly fifty console Ctrl-Cs, and a full `pytest -q` died with exit 137
    the moment it reached tests/test_model_server.py.

    psutil is a core dependency (pyproject) and answers the question by
    looking, not by poking.
    """
    try:
        import psutil
    except ImportError:
        # Without psutil we cannot check safely, and guessing "dead" would
        # delete a live server's pidfile. Say alive; the terminate path is
        # addressed to one pid and is safe on both platforms.
        return True
    return psutil.pid_exists(pid)


def _pid_is_our_server(pid: int) -> bool:
    """Is this pid alive AND actually an ovms process?

    Separate from _pid_is_running on purpose. That one answers a general
    liveness question and has other callers; THIS one is the question to ask
    before signalling, because existence is not identity.

    A pidfile records a NUMBER. Once the process it named has gone, the OS is
    free to hand that number to something else -- explorer.exe, svchost, a
    database. Acting on existence alone therefore pointed `ovat serve --stop`
    at a stranger and killed it.

    Unknown answers are treated as "not ours": without psutil, or when the
    process refuses inspection, the safe move is to not signal.
    """
    try:
        import psutil
    except ImportError:
        return False
    if not psutil.pid_exists(pid):
        return False
    try:
        return "ovms" in psutil.Process(pid).name().lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def stop_from_pidfile(pid_path: str = DEFAULT_PID_PATH, wait_seconds: float = 10.0) -> str:
    """Stop an OVMS recorded in a pidfile; return a short human message.

    This is for the CLI: the process that STARTED the server is long gone, so
    all we have is the pid number on disk. Same escalation as stop(): SIGTERM,
    give it wait_seconds to exit, then SIGKILL. Every odd state (no file, pid
    already dead) is reported as a message, not an exception; stopping an
    already-stopped server is a boring success, not an error.
    """
    if not os.path.exists(pid_path):
        return f"No pidfile at {pid_path}; nothing to stop (server not started here?)."
    try:
        pid = int(open(pid_path, encoding="utf-8").read().strip())
    except ValueError:
        os.remove(pid_path)
        return f"Pidfile {pid_path} was corrupt; removed it."

    def _alive() -> bool:
        return _pid_is_running(pid)

    if not _alive():
        os.remove(pid_path)
        return f"OVMS (pid {pid}) is not running; removed the stale pidfile."

    # Alive is not enough: confirm it is OURS before signalling. If the machine
    # crashed, OVMS died and the OS recycled that number, this pid now belongs
    # to some unrelated program -- and killing it would be our bug, on their
    # process. Refuse rather than guess, and say why.
    if not _pid_is_our_server(pid):
        os.remove(pid_path)
        return (f"pid {pid} is running but is not an ovms process, so the "
                f"pidfile is stale and something else now owns that number. "
                f"Removed the pidfile and left that process alone.")

    os.kill(pid, signal.SIGTERM)    # ask politely first
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if not _alive():
            os.remove(pid_path)
            return f"Stopped OVMS (pid {pid})."
        time.sleep(0.2)
    # Windows has no SIGKILL (its SIGTERM already TerminateProcess-es), so
    # fall back to SIGTERM there; on Linux/macOS this is the real force-kill.
    os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    os.remove(pid_path)
    return f"OVMS (pid {pid}) ignored SIGTERM for {wait_seconds:.0f}s; killed it."
