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
import subprocess
import time
import urllib.error
import urllib.request


class ModelServer:
    """Manages the OVMS process: start, wait-until-ready, stop."""

    def __init__(self, model_name: str, source_model: str | None = None,
                 model_repository_path: str = "models", device: str = "CPU",
                 port: int = 8000, tool_parser: str = "hermes3",
                 reasoning_parser: str | None = None,
                 task: str = "text_generation"):
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
        self.process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v3"

    @property
    def health_url(self) -> str:
        return f"http://localhost:{self.port}/v2/health/ready"

    def start(self) -> None:
        """Launch OVMS in the background. (Flags illustrative -> match them to
        our OVMS version / the demo README.)"""
        # Note to myself: the three flags that matter most here are
        # model_repository_path, source_model and task. Without a model to load,
        # OVMS starts and exits in under a second because it has nothing to do.
        cmd = [
            "ovms",
            "--rest_port", str(self.port),
            "--model_repository_path", self.model_repository_path,
            "--model_name", self.model_name,
            "--task", self.task,
            "--target_device", self.device,
            "--tool_parser", self.tool_parser,
            "--enable_prefix_caching", "true",
        ]
        if self.reasoning_parser:
            cmd += ["--reasoning_parser", self.reasoning_parser]
        # I only add source_model when I have one. On the first run OVMS uses it
        # to download the model, about 5 GB, then later runs see it on disk.
        if self.source_model:
            cmd += ["--source_model", self.source_model]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def wait_until_ready(self, timeout: int = 120) -> bool:
        """Poll the health endpoint until OVMS is up or we time out."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                with urllib.request.urlopen(self.health_url, timeout=2) as r:
                    if r.status == 200:
                        return True
            except (urllib.error.URLError, OSError):
                pass  # not up yet, keep waiting
            time.sleep(2)
        return False

    def stop(self) -> None:
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=10)
            self.process = None
