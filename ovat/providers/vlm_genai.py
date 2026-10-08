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
        from ovat.core.model_scout import identify_model

        self.pipe = ov_genai.VLMPipeline(model_path, device,
                                         **_precision_properties(device))
        self.max_new_tokens = max_new_tokens
        # A unified export (Qwen3.5) is a THINKING model: its chat template
        # opens a reasoning block, and the description only comes after it.
        self.is_unified = identify_model(model_path)[0] == "unified"

    def generate(self, prompt: str, images: list[str]) -> str:
        tensors = [self._load_image(p) for p in images]   # paths -> tensors
        if self.is_unified:
            # Thinking OFF at the source, through the template's own switch
            # (the way GenAILLMProvider passes model.enable_thinking).
            # Measured on the AI PC, Qwen3.5-4B on the GPU, three describe
            # calls: with thinking on, all three spent the 200-token cap on
            # reasoning ("The user wants a description... 1. Identify") and
            # returned no description; letting the reasoning finish took
            # 255-452 tokens and 8.3-13.9 s before a closing </think>. Off,
            # the description came in 13-138 tokens and 0.7-4.3 s. Stripping
            # the reasoning afterwards (text.strip_thinking) only works once
            # it has finished, so it would need that larger budget every call.
            # A ChatHistory is stateless: no start_chat() around it.
            history = ov_genai.ChatHistory([{"role": "user", "content": prompt}])
            history.set_extra_context({"enable_thinking": False})
            return str(self.pipe.generate(history, images=tensors,
                                          max_new_tokens=self.max_new_tokens))
        # start_chat() applies the model's chat template, which gives clean
        # output and a proper stop. (It does NOT cure "!!!!": that is the f16
        # precision problem, see _precision_properties.)
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
        with Image.open(path) as img:
            arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return ov.Tensor(arr)


def _precision_properties(device: str, core=None) -> dict:
    """Pipeline properties that keep a CPU run in f32 where it would not be.

    On an ARM CPU (Apple Silicon) OpenVINO's CPU plugin defaults
    INFERENCE_PRECISION_HINT to f16. Qwen2-VL overflows in f16 and, a few
    tokens in, every logit is NaN, so greedy picks token 0 forever -- which in
    Qwen's vocabulary is "!". Measured on an M-series Mac: "The image features
    the logo!!!!!!..." at the default, and "The image features the logo of
    OpenVINO, which is a toolkit for AI models." with f32 (about 2.8x slower,
    58.6 s vs 21.2 s for 60 tokens, but the fast answer was garbage).

    Only when the CPU's own default is f16, so an x86 CPU (f32 already, or
    bf16 on hardware that has it) keeps exactly what it had: those were never
    measured broken. GPU and NPU are left alone for the same reason.
    """
    if (device or "").upper() != "CPU":
        return {}
    try:
        core = core or ov.Core()
        default = core.get_property("CPU", "INFERENCE_PRECISION_HINT")
    except Exception:                  # a property it cannot read: change nothing
        return {}
    if default == ov.Type.f16:
        return {"INFERENCE_PRECISION_HINT": "f32"}
    return {}
