# ovat/providers/base.py
"""Abstract provider contracts for LLMs, embeddings, retrievers, and VLMs.

All concrete backends (OVMS, OpenVINO GenAI, SQLite-vec, in-memory) implement
these interfaces so workflow YAML configuration selects implementations by
name without changing agent or CLI code.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Interface for text/tool-calling LLM chat completion backends."""

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send messages and optional tool schemas; return normalized response.

        The returned dict must expose ``finish_reason`` and any ``tool_calls``.
        The agent loop dispatches on payload presence (whether ``tool_calls``
        are returned) rather than ``finish_reason``, since NPU serving may
        report ``"stop"`` even when a tool call was decoded.
        """
        ...


class EmbeddingsProvider(ABC):
    """Interface for converting text strings into dense embedding vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class RetrieverProvider(ABC):
    """Interface for storing text chunks with vectors and running similarity search."""

    @abstractmethod
    def add(self, texts: list[str], sources: list[str] | None = None) -> None:
        """Store texts and optional parallel source labels for citations."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        ...

    def close(self) -> None:
        """Release any underlying database connections or file handles.

        Implemented as an optional no-op default because in-memory backends
        hold no external resources, while persistent stores such as sqlite-vec
        override it to close connections cleanly.
        """
        return None


class VLMProvider(ABC):
    """Interface for vision-language models mapping prompt + image paths to text."""

    @abstractmethod
    def generate(self, prompt: str, images: list[str]) -> str:
        ...
