# ovat/providers/embeddings_ovms.py
"""Layer 4: concrete Embeddings plug via OVMS /v3/embeddings.

Same job as GenAIEmbeddingsProvider, but through the OVMS server instead of a
local pipeline. Needs OVMS running. OpenAI-compatible, so it's the OpenAI SDK
pointed at localhost. So, this is the server path for embeddings.
"""
from openai import OpenAI

from ovat.providers.base import EmbeddingsProvider


class OVMSEmbeddingsProvider(EmbeddingsProvider):
    """Embeddings via OVMS /v3/embeddings (OpenAI-compatible)."""

    def __init__(self, base_url: str = "http://localhost:8000/v3",
                 model: str = "bge-small-en-v1.5"):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]
