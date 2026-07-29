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
            "--tool_parser", self.tool_parser,
        ]
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
        # When launching by full path, prepend the binary's folder to the
        # child's PATH so OVMS finds its own DLLs/.so files; this is what
        # setupvars.bat does, done for the user automatically.
        env = dict(os.environ)
        binary_dir = os.path.dirname(os.path.abspath(self.binary))
        if os.path.isdir(binary_dir):
            env["PATH"] = binary_dir + os.pathsep + env.get("PATH", "")
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

    def wait_until_ready(self, timeout: int = 120) -> bool:
        """Poll the health endpoint until OVMS is up or we time out."""
        start = time.time()
        while time.time() - start < timeout:
            # If OVMS already exited, stop waiting the full timeout. The reason
            # is in the log file (self.log_path) instead of being hidden.
            if self.process is not None and self.process.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(self.health_url, timeout=2) as r:
                    if r.status == 200:
                        return True
            except (urllib.error.URLError, OSError):
                pass  # not up yet, keep waiting
            time.sleep(2)
        return False

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
