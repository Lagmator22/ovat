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


#: Default ceiling on one reply. Mirrored by ModelConfig.max_tokens, and a
#: test asserts the two agree -- see tests/test_config.py.
#:
#: It lives HERE as well as in the config because the config is not the only
#: door. Anything constructing this provider directly used to get an unbounded
#: generation, and tests/test_ovms_live.py does exactly that: measured, the KV
#: cache climbed monotonically to 13.6 GB over an hour on a single request
#: that was never going to stop. A safe default belongs at the plug that talks
#: to the server, not only at the config that usually configures it.
DEFAULT_MAX_TOKENS = 4096


class OVMSLLMProvider(LLMProvider):
    """Talks to OVMS /v3 using the OpenAI SDK (as OVMS is OpenAI-compatible)."""

    def __init__(self, base_url: str = "http://localhost:8000/v3",
                 model: str = "Qwen3-8B-int4-ov", timeout: float = 120.0,
                 max_tokens: int | None = DEFAULT_MAX_TOKENS,
                 temperature: float = 0.0):
        # api_key is required by the SDK but ignored by OVMS, any string works.
        # timeout caps each request; the SDK default (~600s) would freeze the
        # CLI for ten minutes on a hung server before erroring.
        self.client = OpenAI(base_url=base_url, api_key="not-needed",
                             timeout=timeout)
        self.model = model
        # None still means "do not send the field, let the server decide",
        # but it is no longer the DEFAULT: see DEFAULT_MAX_TOKENS above for
        # what an unbounded default actually cost. Read fresh on every call,
        # so the TUI can retune it live.
        self.max_tokens = max_tokens
        # Sent on every request. It comes from model.temperature so all four
        # engines sample identically; two of them used to hardcode 0 while
        # this one took the server default, which made cross-engine timings
        # compare determinism against sampling rather than framework against
        # framework.
        self.temperature = temperature

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        extra = {} if self.max_tokens is None else {"max_tokens": self.max_tokens}
        extra["temperature"] = self.temperature
        try:
            return self._ask(messages, tools, extra)
        except Exception as exc:
            raise self._explain(exc) from exc

    def _explain(self, exc: Exception) -> Exception:
        """Turn OVMS's "Invalid request URL" into the sentence behind it.

        OVMS serves its OpenAI API under /v3. A base_url ending in /v1 is
        plano's listener path, not OVMS's, so pointing a client at
        localhost:8000/v1 with no gateway running reaches OVMS and gets

            Error code: 400 - {'error': 'Invalid request URL'}

        which names neither the path nor the fix. This is easy to hit because
        the plano example config legitimately uses /v1 -- copy it, run without
        plano, and the server is healthy while every request 400s.
        """
        text = str(exc)
        # The NPU prompt cap, which arrives as a Mediapipe graph failure and
        # names nothing a user can act on:
        #
        #   Mediapipe execution failed. MP status - INVALID_ARGUMENT:
        #   CalculatorGraph::Run() failed: Calculator::Process() for node
        #   "LLMExecutor" failed: Input length exceeds the maximum allowed
        #   length
        #
        # OVMS caps an NPU prompt at 1024 tokens unless --max_prompt_len says
        # otherwise, and an agent turn is a system prompt plus every tool
        # schema plus the history, which passes 1024 almost immediately. The
        # server is healthy and the model is fine; the cap is simply too low.
        if "Input length exceeds the maximum allowed length" in text:
            return RuntimeError(
                "the prompt was longer than this server allows.\n"
                "On NPU, OVMS caps the prompt at 1024 tokens unless it is "
                "started with --max_prompt_len, and one agent turn -- system "
                "prompt + every tool schema + the history -- passes that "
                "almost immediately.\n"
                "Fix: set model.ovms_max_prompt_len in the workflow (4096 is "
                "a reasonable start) and restart the server. Nothing is wrong "
                "with the model or the server; the cap is too low.\n"
                "Shortening the question will not help much: the tool "
                "schemas are most of the prompt.")
        if "Invalid request URL" not in text and "404" not in text:
            return exc
        url = str(getattr(self.client, "base_url", "")).rstrip("/")
        if url.endswith("/v1"):
            return RuntimeError(
                f"OVMS rejected the request URL. model.ovms_url ends in /v1, "
                f"which is the plano gateway's path -- OVMS itself serves its "
                f"OpenAI API under /v3.\n"
                f"  now:  {url}\n"
                f"  try:  {url[:-3]}/v3\n"
                f"Use /v1 only when plano is actually running in front of "
                f"OVMS; otherwise point straight at /v3.")
        if not url.endswith("/v3"):
            return RuntimeError(
                f"OVMS rejected the request URL: {text}\n"
                f"model.ovms_url is {url!r}; OVMS serves its OpenAI API under "
                f"/v3, so the URL normally ends there.")
        return exc

    def _ask(self, messages: list[dict], tools: list[dict] | None,
             extra: dict) -> dict:
        response = self.client.chat.completions.create(
            model=self.model, #model name is pulled from the YAML config file
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            **extra,
        )
        choice = response.choices[0]
        # OVMS reports token usage on every response; keep it instead of
        # dropping it; the agent loop turns it into the run trace (Layer 7).
        usage = getattr(response, "usage", None)
        return {
            "finish_reason": choice.finish_reason,    # "stop" or "tool_calls"
            "content": choice.message.content,
            "tool_calls": choice.message.tool_calls,  # None or a list
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            } if usage is not None else None,
            "raw": response,
        }
