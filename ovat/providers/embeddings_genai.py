# ovat/providers/embeddings_genai.py
"""Local embeddings provider backed by ``openvino_genai.TextEmbeddingPipeline``."""
import openvino_genai as ov_genai

from ovat.providers.base import EmbeddingsProvider


class GenAIEmbeddingsProvider(EmbeddingsProvider):
    """Local text embeddings via openvino_genai.TextEmbeddingPipeline."""

    def __init__(self, model_path: str, device: str = "CPU"):
        self.pipe = ov_genai.TextEmbeddingPipeline(model_path, device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # One vector per input text. (bge-small -> 384 floats each.)
        return self.pipe.embed_documents(texts)
