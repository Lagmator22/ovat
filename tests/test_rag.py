# tests/test_rag.py
"""Tests for the RAG path: chunking, indexing, retrieval, and provider choice.

Note to myself: I never load a real embedding model here. I inject a tiny
FakeEmbedder that maps text to a deterministic vector. That is enough to prove
the whole pipeline (chunk -> store -> nearest-neighbour -> citation) works,
and it runs in milliseconds on the Mac with nothing downloaded. The only piece
this cannot cover is building the real genai embedder, which is a live test on
the AI PC.
"""
import pytest

from ovat.agent.factory import build_embedder, build_rag, build_retriever
from ovat.config.workflow import WorkflowConfig
from ovat.providers.base import RetrieverProvider
from ovat.providers.retriever_sqlitevec import SQLiteVecRetrieverProvider
from ovat.rag.indexer import chunk_text, index_folder
from ovat.tools.search_docs import search_docs_impl


class FakeEmbedder:
    """Deterministic stand-in for a real embedder.

    Same text in -> same vector out, and different text -> a different vector.
    That is all the retriever needs: an exact-match query lands distance 0 on
    the chunk it came from, so I can assert which chunk (and source) comes back.
    """

    def embed(self, texts):
        vectors = []
        for t in texts:
            base = sum(ord(c) for c in t)
            vectors.append([float((base + i) % 97) for i in range(384)])
        return vectors


def _rag_cfg(emb_provider="genai", ret_provider="sqlite-vec"):
    return WorkflowConfig(
        model={"name": "m"},
        rag={
            "embeddings": {"provider": emb_provider, "dim": 384},
            "retriever": {"provider": ret_provider, "db_path": ":memory:"},
        },
    )


# chunk_text

def test_chunk_text_splits_with_overlap():
    # step = size - overlap = 3, so windows start at 0, 3, 6, 9.
    assert chunk_text("abcdefghij", size=4, overlap=1) == ["abcd", "defg", "ghij", "j"]


def test_chunk_text_empty_input_returns_no_chunks():
    assert chunk_text("    ", size=10, overlap=2) == []


def test_chunk_text_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", size=4, overlap=4)


def test_chunk_text_rejects_non_positive_size():
    with pytest.raises(ValueError):
        chunk_text("abc", size=0)


# index_folder + search_docs (end to end with a fake embedder)

def test_index_folder_then_search_returns_chunk_with_citation(tmp_path):
    (tmp_path / "a.md").write_text("python is great for ai", encoding="utf-8")
    (tmp_path / "b.txt").write_text("rust is fast and safe", encoding="utf-8")
    # A file type I do not index, to prove the filter works.
    (tmp_path / "ignore.bin").write_text("binary junk", encoding="utf-8")

    retriever = SQLiteVecRetrieverProvider(FakeEmbedder(), dim=384, db_path=":memory:")
    summary = index_folder(str(tmp_path), retriever, size=512, overlap=0)
    assert summary == {"files": 2, "chunks": 2}

    hits = search_docs_impl("python is great for ai", top_k=1, retriever=retriever)
    assert hits[0]["text"] == "python is great for ai"
    assert hits[0]["source"].endswith("a.md")        # the citation half of RAG


def test_index_folder_missing_folder_raises():
    retriever = SQLiteVecRetrieverProvider(FakeEmbedder(), dim=384, db_path=":memory:")
    with pytest.raises(FileNotFoundError):
        index_folder("/no/such/folder", retriever)


# provider selection (the ABC swap)

def test_build_retriever_builds_sqlitevec_with_injected_embedder():
    retriever = build_retriever(_rag_cfg(), FakeEmbedder())
    assert isinstance(retriever, SQLiteVecRetrieverProvider)
    retriever.add(["hello world"], sources=["x.md"])
    assert retriever.retrieve("hello world", top_k=1)[0]["source"] == "x.md"


def test_build_embedder_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown embeddings provider"):
        build_embedder(_rag_cfg(emb_provider="nope"))


def test_build_retriever_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown retriever provider"):
        build_retriever(_rag_cfg(ret_provider="nope"), FakeEmbedder())


def test_build_rag_returns_none_when_no_rag_section():
    assert build_rag(WorkflowConfig(model={"name": "m"})) is None


# close() — the retriever must release its SQLite connection

def test_close_releases_the_connection_and_is_idempotent():
    retriever = SQLiteVecRetrieverProvider(FakeEmbedder(), dim=384, db_path=":memory:")
    retriever.add(["hello"], sources=["x.md"])
    retriever.close()
    # After close the handle is gone, so any use must fail loudly...
    with pytest.raises(AttributeError):
        retriever.retrieve("hello")
    # ...but closing again is safe (idempotent), like a guarded delete.
    retriever.close()


def test_context_manager_closes_on_exit(tmp_path):
    db = str(tmp_path / "idx.db")
    with SQLiteVecRetrieverProvider(FakeEmbedder(), dim=384, db_path=db) as retriever:
        retriever.add(["persist me"], sources=["a.md"])
    assert retriever.db is None                      # __exit__ really closed it
    # The data survived the close: a fresh open finds the chunk on disk.
    with SQLiteVecRetrieverProvider(FakeEmbedder(), dim=384, db_path=db) as reopened:
        assert reopened.retrieve("persist me", top_k=1)[0]["source"] == "a.md"


def test_base_retriever_close_is_a_safe_default_noop():
    # Any RetrieverProvider can be close()d, even ones that hold nothing.
    class NothingRetriever(RetrieverProvider):
        def add(self, texts, sources=None): ...
        def retrieve(self, query, top_k=5): return []

    NothingRetriever().close()                       # must not raise
