# ovat/providers/base.py
"""
Layer 4: Provider Abstraction (the contracts).

An ABC (Abstract Base Class) is a *contract*: it says "any LLM provider MUST
have a .chat() method" without saying HOW. We then write multiple concrete
implementations (OVMS, GenAI parts of all sockets/contracts) that all honor the contract.
The YAML config picks one by name, so swapping a backend = changing a string in config, not
rewriting code. This is the single design idea the whole toolkit leans on. Rest of Layer 4 
is just to fill in the blanks for each backend.

TLDR:
Socket + plug = this is the whole toolkit's big idea for ABCs, base.py says "must have chat()"; 
llm_genai.py does chat() one way (local), and later llm_ovms.py does it another way (server),
same chat(), same return dict. So the rest of the code can call .chat() without caring which
engine runs underneath.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Contract/Socket for anything that can run an LLM chat completion."""
    """Any LLM provider in OVAT must have a chat() method, MUST exist"""
    @abstractmethod # like pure virtual = 0 in cpp 
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send messages (+ optional tools), return the raw response.

        The returned dict MUST expose `finish_reason` and any `tool_calls`.
        The agent loop (Layer 3) dispatches on the PAYLOAD:
          - tool_calls present  -> run the tools, append results, loop
          - tool_calls absent   -> return the content as the answer
        finish_reason is recorded and used only to diagnose a broken reply
        ("tool_calls" with none attached, or "length" for a cut-off one),
        because OVMS documents NPU serving labelling a real call "stop".
        """
        ...


class EmbeddingsProvider(ABC):
    """Contract/Socket for turning text into vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class RetrieverProvider(ABC):
    """Contract/Socket for storing vectors and searching them by similarity."""

    @abstractmethod
    def add(self, texts: list[str], sources: list[str] | None = None) -> None:
        """Store these texts. `sources` is optional and lines up by index, so a
        later retrieve() can say which file each chunk came from (citations)."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        ...

    def close(self) -> None:
        """Release whatever the retriever holds (DB connections, files).

        NOT abstract on purpose: a plain virtual with an empty default body,
        not pure virtual. In-memory retrievers hold nothing, so they inherit
        this no-op; sqlite-vec overrides it to close its connection. Callers
        can always call close() without caring which backend they got.
        """
        return None


class VLMProvider(ABC): # adds the support for multimodality supporting models like gemma 4[OpenVINO formats are experimental] and no video support yet
    """Contract/Socket for a vision-language model: a text prompt + images -> text.

    This is the multimodal contract/socket (Qwen2-VL today, Gemma 4 / Qwen3-VL later).
    Concrete plugs handle loading the image files into tensors; callers just
    pass file paths.
    """

    @abstractmethod
    def generate(self, prompt: str, images: list[str]) -> str:
        ...
