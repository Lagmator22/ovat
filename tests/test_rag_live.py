# tests/test_rag_live.py
"""Real-model RAG test: the actual bge-small embedder, not a fake.

Note to myself: this needs the OpenVINO bge-small model exported to disk under
models/bge-small-en-v1.5 (the same folder examples/workflow.yml points at). If
that folder is missing, the test skips, so CI and a fresh clone stay green. On
my Mac and on the AI PC, where the model exists, it runs for real.

What it proves that the fake-embedder tests cannot: the genai embedder loads,
produces vectors whose distances are meaningful, and retrieval ranks documents
by MEANING. Each query below uses words the matching document does not contain.
"""
import os

import pytest

from ovat.agent.factory import build_rag
from ovat.config.workflow import load_workflow
from ovat.rag.indexer import index_folder
from ovat.tools.search_docs import search_docs_impl

MODEL_DIR = "models/bge-small-en-v1.5"

needs_model = pytest.mark.skipif(
    not os.path.isdir(MODEL_DIR),
    reason=f"embedding model not found at {MODEL_DIR}; export it first",
)

# Mark the file so `pytest -m "not rag"` can skip the heavier model load on demand.
pytestmark = pytest.mark.rag


@needs_model
def test_real_bge_small_ranks_documents_by_meaning(tmp_path):
    cfg = load_workflow("examples/workflow.yml")
    cfg.rag.embeddings.model = MODEL_DIR        # be explicit, do not rely on default
    cfg.rag.retriever.db_path = ":memory:"      # self-contained, leaves no file

    (tmp_path / "finance.md").write_text(
        "Our Q3 revenue grew 12 percent driven by cloud subscriptions. "
        "Operating margin improved as data-center costs fell.", encoding="utf-8")
    (tmp_path / "space.md").write_text(
        "The James Webb Space Telescope observes infrared light to study "
        "the earliest galaxies formed after the Big Bang.", encoding="utf-8")
    (tmp_path / "cooking.md").write_text(
        "To bake sourdough bread you need flour, water, salt and a live "
        "starter, then let the dough ferment overnight.", encoding="utf-8")

    retriever = build_rag(cfg)
    summary = index_folder(str(tmp_path), retriever,
                           size=cfg.rag.chunk.size, overlap=cfg.rag.chunk.overlap)
    assert summary["files"] == 3

    # Each query avoids the literal words of its target document on purpose.
    cases = {
        "how did the company earnings perform last quarter?": "finance.md",
        "what telescope looks at the early universe?": "space.md",
        "a recipe that uses a fermented starter": "cooking.md",
    }
    for query, expected in cases.items():
        hits = search_docs_impl(query, top_k=1, retriever=retriever)
        assert hits, f"no result for: {query}"
        assert hits[0]["source"].endswith(expected), (
            f"{query!r} matched {hits[0]['source']}, expected {expected}")
        # The citation contract: every hit carries text + source + distance.
        assert {"text", "source", "distance"} <= set(hits[0])
