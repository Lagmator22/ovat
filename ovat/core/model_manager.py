# ovat/core/model_manager.py
"""Layer 8: thin Python wrapper over the OVMS model-management CLI.

Connects & shell out to the `ovms` binary (which lives on Linux/Windows or inside
the OVMS Docker container for the typical setup interaction with pull and list_models etc commands).
On a Mac with no native OVMS these will raise FileNotFoundError,
which is expected. The point is a clean Python API over the CLI so the rest of
the toolkit never builds command strings by hand. Basically easy access to the ovms cli.
"""
import subprocess

class ModelManager:
    """Wraps OVMS model-management CLI commands (--pull, --list_models)."""

    # How long `--list_models` may take before we call the binary wedged.
    # Listing a repository is a directory read: if it has not answered in half
    # a minute it is not going to. `pull` deliberately has NO timeout, see the
    # note on that method.
    LIST_TIMEOUT_S = 30.0

    def __init__(self, ovms_binary: str = "ovms"):
        self.ovms = ovms_binary

    def pull(self, source_model: str, model_repository_path: str = "models") -> str:
        """Download a model, e.g. 'OpenVINO/Qwen3-8B-int4-ov'.

        model_repository_path is where OVMS puts it, and OVMS needs telling:
        without the flag it has no repository to write into and exits non-zero.

        DELIBERATELY UNBOUNDED. An 8B model is several gigabytes over whatever
        connection the machine has; half an hour is a normal pull, not a hang.
        A timeout here would abort real downloads and leave a half-written
        repository behind, which is worse than waiting. The asymmetry with
        list_models below is on purpose, not an oversight.
        """
        result = subprocess.run(
            [self.ovms, "--pull", "--source_model", source_model,
             "--model_repository_path", model_repository_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Model pull failed: {result.stderr.strip()}")
        return result.stdout

    def list_models(self, model_repository_path: str = "models") -> list[str]:
        """What OVMS can serve out of `model_repository_path`.

        The flag is not optional. `ovms --list_models` on its own has no
        repository to list and exits non-zero, which is why `ovat models list`
        never worked: the RuntimeError below escaped as a raw traceback.

        Bounded, unlike pull(). This only reads a directory, so a binary that
        has not answered in LIST_TIMEOUT_S is wedged. Without the bound
        `ovat models list` froze with no output and no explanation, which is
        indistinguishable from the command being broken.
        """
        try:
            result = subprocess.run(
                [self.ovms, "--list_models",
                 "--model_repository_path", model_repository_path],
                capture_output=True, text=True, errors="replace",
                timeout=self.LIST_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"ovms did not answer --list_models within "
                f"{self.LIST_TIMEOUT_S:.0f}s. The binary at {self.ovms} may be "
                f"wedged; try running it by hand to see what it is waiting on."
            ) from None
        if result.returncode != 0:
            raise RuntimeError(f"List models failed: {result.stderr.strip()}")
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
