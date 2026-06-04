# ovat/providers/vlm_genai.py
"""Layer 4: concrete VLM (vision-language but not video playback yet) plug: image(s) + text -> text.

Fills the VLMProvider socket using openvino_genai.VLMPipeline (Qwen2-VL today;
Gemma 4(no vid) / Qwen3-VL later once stable). Runs on Mac CPU.

The one new idea vs the text plugs: the model can't read a .jpg file, it needs
the pixels as a tensor. So _load_image does file -> pixel grid -> ov.Tensor.
An image is height x width x 3 numbers (R,G,B per pixel); a Tensor is just
OpenVINO's N-dimensional array (like a multi-dimensional std::vector).
"""
import numpy as np
from PIL import Image
import openvino as ov
import openvino_genai as ov_genai

from ovat.providers.base import VLMProvider


class GenAIVLMProvider(VLMProvider):
    """Local vision-language model via openvino_genai.VLMPipeline."""

    def __init__(self, model_path: str, device: str = "CPU", max_new_tokens: int = 200):
        self.pipe = ov_genai.VLMPipeline(model_path, device)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, images: list[str]) -> str:
        tensors = [self._load_image(p) for p in images]   # paths -> tensors
        # start_chat() applies the model's chat template, which gives clean
        # output and a proper stop. Without it, some models ramble ("!!!!").
        self.pipe.start_chat()
        try:
            result = self.pipe.generate(
                prompt, images=tensors, max_new_tokens=self.max_new_tokens
            )
        finally:
            self.pipe.finish_chat()
        return str(result)

    @staticmethod
    def _load_image(path: str) -> ov.Tensor:
        img = Image.open(path).convert("RGB")     # force 3 colour channels
        arr = np.array(img, dtype=np.uint8)       # shape (height, width, 3)
        return ov.Tensor(arr)                     # wrap as an OpenVINO tensor
