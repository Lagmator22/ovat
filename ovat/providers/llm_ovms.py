# ovat/providers/llm_ovms.py
"""Layer 4: concrete LLM plug that talks to the OVMS server.

Fills the LLMProvider socket by calling OVMS's OpenAI-compatible /v3 endpoint
with the OpenAI SDK. Unlike GenAILLMProvider (which runs a local pipeline),
this one needs an OVMS server running (locally via Docker, or on the Intel AI
PC). Because OVMS supports --tool_parser, THIS plug is the one that does real
tool calling and so chat() can return finish_reason="tool_calls". So this is the 
server path for llms.
"""
from openai import OpenAI

from ovat.providers.base import LLMProvider


class OVMSLLMProvider(LLMProvider):
    """Talks to OVMS /v3 using the OpenAI SDK (as OVMS is OpenAI-compatible)."""

    def __init__(self, base_url: str = "http://localhost:8000/v3",
                 model: str = "Qwen3-8B-int4-ov"):
        # api_key is required by the SDK but ignored by OVMS, any string works.
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        response = self.client.chat.completions.create(
            model=self.model, #model name is pulled from the YAML config file
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
        )
        choice = response.choices[0]
        return {
            "finish_reason": choice.finish_reason,    # "stop" or "tool_calls"
            "content": choice.message.content,
            "tool_calls": choice.message.tool_calls,  # None or a list
            "raw": response,
        }
