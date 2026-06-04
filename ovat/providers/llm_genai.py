# ovat/providers/llm_genai.py
"""Layer 4: concrete LLM plug that runs a text model on this machine.

GenAILLMProvider fills the LLMProvider socket using openvino_genai.LLMPipeline.
No server needed. Runs natively on Mac CPU. This is the "GenAI fallback"
for local dev (the OVMS plug, OVMSLLMProvider, comes later for the server path).

Limitation: since direct GenAI/LLMs does NOT do OVMS-style tool calling, so chat()
here always returns finish_reason="stop" and tool_calls=None. It's for real
text generation + wiring; true tool-calling agents use the OVMS plug.
"""
import openvino_genai as ov_genai

from ovat.providers.base import LLMProvider


class GenAILLMProvider(LLMProvider): # obey the LLMprovider rulebook
    """Runs a local text LLM via openvino_genai.LLMPipeline."""

    def __init__(self, model_path: str, device: str = "CPU", max_new_tokens: int = 256):
        """
        init (Constructor) runs once here to load the converted model onto a device. 
        This is the same call made in [PoC] OvaSearch's C++:
        ov::genai::LLMPipeline pipe(path, "CPU"). CPU is default path.
        'self' in Python == 'this' keyword in C++ 
        """
        self.pipe = ov_genai.LLMPipeline(model_path, device)
        self.max_new_tokens = max_new_tokens #to remember/use the max_new_tokens limit for later

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        prompt = self._format(messages)
        text = self.pipe.generate(prompt, max_new_tokens=self.max_new_tokens)
        return {
            "finish_reason": "stop", # as genai directly can't request tools so writes
            "content": str(text),    # text only then stops
            "tool_calls": None,
            "raw": text,
        }

    @staticmethod # this helper doesn't need self, it's a plain function that lives in the class
    def _format(messages: list[dict]) -> str:
        """
        Flatten [{role, content}, ...] into one prompt string, then cue
        the assistant to answer. (Simple for now; we can swap in 
        the model's real chat template later and the chat() interface won't change.)
        """
        lines = [f"{m['role']}: {m['content']}" for m in messages] # make the string role: content, and collect them all into a list called lines.
        lines.append("assistant:") # add assistant prompt so the LLM knows to start generating a response.
        return "\n".join(lines) # glues all the strings in 'lines' together with "\n" in between them and returns a single string.  
