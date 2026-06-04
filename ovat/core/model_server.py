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

    def __init__(self, model_name: str, device: str = "CPU",
                 port: int = 8000, tool_parser: str = "hermes3"):
        self.model_name = model_name
        self.device = device
        self.port = port
        self.tool_parser = tool_parser
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
        cmd = [
            "ovms",
            "--rest_port", str(self.port),
            "--model_name", self.model_name,
            "--target_device", self.device,
            "--tool_parser", self.tool_parser,
            "--enable_prefix_caching",
        ]
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
