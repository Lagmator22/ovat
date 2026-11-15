# ovat/core/device_manager.py
"""OpenVINO Runtime hardware detection and device recommendation.

Queries OpenVINO for available compute devices (``CPU``, ``GPU``, ``NPU``)
and selects default target devices per model workload.
"""
import openvino as ov

class DeviceManager:
    """Detects available hardware and recommends a device per model type.

    Default routing rules:
      - Embeddings -> NPU if available (small, static-shape, low power).
      - LLM        -> GPU if available. While OVMS supports NPU text generation
        and tool calling with channel-wise symmetric INT4 models
        (``-int4-cw-ov``), stock group-quantised ``-int4-ov`` exports fail NPU
        compilation, and NPU limits batching, beam search, and prompt length
        (``--max_prompt_len`` defaults to 1024). GPU works across any export.
      - Whisper    -> CPU (small model, CPU latency is acceptable).
      - Fallback   -> CPU (universally supported).
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
