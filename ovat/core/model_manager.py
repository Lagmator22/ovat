# ovat/core/model_manager.py
"""Layer 8: thin Python wrapper over the OVMS model-management CLI.

These shell out to the `ovms` binary (which lives on Linux/Windows or inside
the OVMS Docker container). On a Mac with no native OVMS these will raise;
that's expected. The point is a clean Python API over the CLI so the rest of
the toolkit never builds command strings by hand.
"""
import subprocess


class ModelManager:
    """Wraps OVMS model-management CLI commands (--pull, --list_models)."""

    def __init__(self, ovms_binary: str = "ovms"):
        self.ovms = ovms_binary

    def pull(self, source_model: str) -> str:
        """Download a model, e.g. 'OpenVINO/Qwen3-8B-int4-ov'."""
        result = subprocess.run(
            [self.ovms, "--pull", "--source_model", source_model],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Model pull failed: {result.stderr}")
        return result.stdout

    def list_models(self) -> list[str]:
        result = subprocess.run(
            [self.ovms, "--list_models"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"List models failed: {result.stderr}")
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
