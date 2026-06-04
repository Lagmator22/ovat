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
      - Embeddings -> NPU if available (small, static-shape, low power). NPU can't do tool calling, 
                      so ONLY embeddings go here.
      - LLM        -> GPU if available (dynamic shapes, KV cache, tool calling).
      - Whisper    -> CPU (small model, CPU latency is fine).
      - Anything   -> CPU as the universal fallback (always works, INT4).
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
