# ovat/core/device_manager.py
"""Layer 9: OpenVINO Runtime / AI PC hardware detection.

Asks openvino what hardware exists and decides which device each model
type should run on. On a Mac only ['CPU']. Runs on an Intel AI PC, will see ['CPU', 'GPU', 'NPU']
and route accordingly. Nothing here changes b/w mac and the AIPC
except what get_available_devices() returns.
"""
import openvino as ov

class DeviceManager:
    """Detects available hardware and recommends a device per model type.

    Routing rules (Layer 9 proposal table):
      - Embeddings -> NPU if available (small, static-shape, low power).
      - LLM        -> GPU if available. NOT because the NPU cannot serve an
        LLM: OVMS supports NPU text generation and its own demo does it, with
        tool calling, on LunarLake -- the same silicon as this project's AI PC.
        The constraint is the EXPORT, and it is strict.

        MEASURED HERE, 2026-08-11: Qwen3.5-4B and Qwen3.5-0.8B both fail
        identically at compilation, [NPU_VCL] Compilation failed (0x78000004),
        before a single token. Both are the stock `-int4-ov` builds, which are
        GROUP quantised. OVMS documents that an NPU model must be exported
        INT4 with `--sym --ratio 1.0 --group-size -1` -- channel-wise,
        symmetric -- and ships a separate family of `-int4-cw-ov` models for
        exactly this. A group-quantised export is not a model the NPU plugin
        can compile, so the failure is the expected answer to the wrong file
        rather than a limit of the device.

        GPU is still the default because it costs nothing to be right on any
        export, while NPU additionally gives up batching, beam search and
        log_probs, and caps the prompt (`--max_prompt_len`, default 1024).
        To serve an agent on NPU, pull a `-int4-cw-ov` model; see the NPU
        section of the README.
      - Whisper    -> CPU (small model, CPU latency is fine).
      - Anything   -> CPU as the universal fallback (always works; low-bit
                      weights keep the model in RAM).
    """

    def __init__(self):
        self.core = ov.Core()
        # On Mac: ['CPU']. On a full AI PC: ['CPU', 'GPU', 'NPU'].
        self.available = self.core.get_available_devices()

    def get_llm_device(self) -> str:
        # LLM needs dynamic shapes + tool calling -> prefer GPU, fall back to CPU.
        return "GPU" if "GPU" in self.available else "CPU"

    def get_embedding_device(self) -> str:
        # Embeddings are static-shape + low-power -> NPU is ideal if present.
        return "NPU" if "NPU" in self.available else "CPU"

    def get_whisper_device(self) -> str:
        # Whisper-base is small; CPU is always acceptable.
        return "CPU"

    def summary(self) -> dict:
        return {
            "available": self.available,
            "llm": self.get_llm_device(),
            "embeddings": self.get_embedding_device(),
            "whisper": self.get_whisper_device(),
        }


if __name__ == "__main__":
    dm = DeviceManager()
    print(dm.summary())
    # On Mac output: {'available': ['CPU'], 'llm': 'CPU', 'embeddings': 'CPU', 'whisper': 'CPU'}
