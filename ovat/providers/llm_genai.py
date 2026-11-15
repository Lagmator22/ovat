# ovat/providers/llm_genai.py
"""Local LLM provider backed by ``openvino_genai``.

Runs inference in-process on the target OpenVINO device (defaulting to CPU)
without an OVMS server. ``chat()`` always returns ``finish_reason="stop"`` and
``tool_calls=None`` because direct GenAI pipelines do not emit OVMS-style tool
calls.
"""
import openvino_genai as ov_genai

from ovat.providers.base import LLMProvider


class GenAILLMProvider(LLMProvider):
    """Runs a local text LLM or unified multimodal export via openvino_genai."""

    def __init__(self, model_path: str, device: str = "CPU",
                 max_new_tokens: int | None = 256,
                 enable_thinking: bool | None = None):
        """Load the OpenVINO GenAI pipeline for ``model_path`` onto ``device``.

        ``max_new_tokens=None`` disables the generation cap (stopping only on
        EOS), while the default of 256 prevents runaway generation if a base
        model or misconfigured template fails to emit EOS.

        Unified multimodal exports (e.g., Qwen3.5) require ``VLMPipeline`` even
        for text-only prompts: constructing ``LLMPipeline`` succeeds but fails
        at generation with a missing ``input_ids`` tensor port error, so the
        pipeline class is selected from ``identify_model(model_path)``.
        """
        # model_scout is the single source of truth for "what kind is this".
        from ovat.core.model_scout import identify_model

        self.is_unified = identify_model(model_path)[0] == "unified"
        if self.is_unified:
            self.pipe = ov_genai.VLMPipeline(model_path, device)
        else:
            self.pipe = ov_genai.LLMPipeline(model_path, device)
        # Read fresh on every chat() call, so a caller can retune the cap on a
        # live provider without paying to rebuild the pipeline.
        self.max_new_tokens = max_new_tokens
        # The chat template's thinking switch (model.enable_thinking). None
        # leaves the template's own default. Fixed for the provider's life,
        # so each instance keeps ONE input type (see chat()).
        self.enable_thinking = enable_thinking

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             on_token=None) -> dict:
        """Generate a reply. With on_token, stream each token as it decodes.

        on_token is optional so non-streaming callers retain existing behavior.
        When supplied, openvino_genai invokes the streamer callback per decoded
        token, forwarding text and returning False to continue generation.
        The final return dict is identical either way.
        """
        # A text LLM gets the conversation AS a conversation. LLMPipeline
        # applies the model's chat template itself, so the old flattened
        # "role: content" string was wrapped as ONE user message holding fake
        # role labels -- verified by rendering it with a real tokenizer.
        #
        # A unified export still takes the flattened string inside
        # start_chat(): that path is verified on the AI PC with Qwen3.5, and
        # switching it is unverified until it is run there. Note the pipeline
        # refuses to MIX the two input types on one instance ("Chat doesn't
        # support switching between input types"), so each path keeps one.
        #
        # The exception is enable_thinking: the switch only exists as chat
        # template context, so a unified model that is GIVEN it goes through
        # a ChatHistory too. Opt-in, and fixed per instance, so one instance
        # still never mixes input types.
        if self.is_unified and self.enable_thinking is None:
            prompt = self._format(messages)
        else:
            prompt = ov_genai.ChatHistory(
                [{"role": m["role"], "content": str(m.get("content") or "")}
                 for m in messages])
            if self.enable_thinking is not None:
                prompt.set_extra_context(
                    {"enable_thinking": self.enable_thinking})
        # Omitting the argument entirely is what "no cap" means to
        # openvino_genai; passing None would not be read as a number.
        limit = ({} if self.max_new_tokens is None
                 else {"max_new_tokens": self.max_new_tokens})
        # VLMPipeline.generate REQUIRES images; there is no text-only overload,
        # so an empty list is how a unified model is asked a text question.
        if self.is_unified:
            limit["images"] = []
        if on_token is None:
            text = self._generate(prompt, **limit)
        else:
            def _streamer(token: str):
                # on_token may return True to STOP generation early (that is
                # openvino_genai's streamer contract); None/False continues.
                return bool(on_token(token))
            text = self._generate(prompt, streamer=_streamer, **limit)
        return {
            "finish_reason": "stop", # as genai directly can't request tools so writes
            "content": str(text),    # text only then stops
            "tool_calls": None,
            "raw": text,
        }

    def _generate(self, prompt, **kwargs):
        """One generate call, with the chat session a unified model needs.

        VLMPipeline needs start_chat()/finish_chat() around generation so the
        model's own chat template is applied. Without it some models ramble
        ("!!!!") instead of stopping, which is why vlm_genai.py does the same.
        finish_chat() runs in a finally: an exception mid-generation must not
        leave the pipeline stuck in a chat session for every later turn.
        """
        # A ChatHistory is the stateless path: the pipeline applies the
        # template itself and needs no chat session around it.
        if not self.is_unified or isinstance(prompt, ov_genai.ChatHistory):
            return self.pipe.generate(prompt, **kwargs)
        self.pipe.start_chat()
        try:
            return self.pipe.generate(prompt, **kwargs)
        finally:
            self.pipe.finish_chat()

    @staticmethod # this helper doesn't need self, it's a plain function that lives in the class
    def _format(messages: list[dict]) -> str:
        """
        Flatten [{role, content}, ...] into one prompt string, then cue
        the assistant to answer. Used ONLY for unified exports now; a text
        LLM gets a ChatHistory instead, so the model's own template sees the
        real roles. See chat().
        """
        lines = [f"{m['role']}: {m['content']}" for m in messages] # make the string role: content, and collect them all into a list called lines.
        lines.append("assistant:") # add assistant prompt so the LLM knows to start generating a response.
        return "\n".join(lines) # glues all the strings in 'lines' together with "\n" in between them and returns a single string.  
