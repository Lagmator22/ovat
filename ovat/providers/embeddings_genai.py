# ovat/providers/embeddings_genai.py
"""Layer 4: concrete Embeddings plug: text -> vectors, locally on CPU.

Fills the EmbeddingsProvider socket using openvino_genai.TextEmbeddingPipeline
(bge-small today).

What an "embedding" is: turning a piece of text into a list of numbers (vectors) 
that captures its meaning. Similar meanings -> similar numbers, which
is what lets us later search by meaning instead of exact keywords. bge-small
produces 384 numbers(or has 384 dimensions) per text. Note to look into more options 
like matryoshka-embedding-model to get better results (768 -> 128 dimensions all on edge).
"""
import openvino_genai as ov_genai

from ovat.providers.base import EmbeddingsProvider


class GenAIEmbeddingsProvider(EmbeddingsProvider):
    """Local text embeddings via openvino_genai.TextEmbeddingPipeline."""

    def __init__(self, model_path: str, device: str = "CPU"):
        self.pipe = ov_genai.TextEmbeddingPipeline(model_path, device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # One vector per input text. (bge-small -> 384 floats each.)
        return self.pipe.embed_documents(texts)
