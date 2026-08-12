# ovat/providers/retriever_memory.py
"""An in-memory vector store: the RetrieverProvider socket with no file.

WHY A SECOND BACKEND EXISTS AT ALL. The architecture doc is blunt that an
abstraction with a single implementation is indirection you pay for and get
nothing back from. `RetrieverProvider` had exactly one, so it was a promise
rather than a demonstration. This is the second, and it cost one file and one
line in the factory precisely because the socket was the right shape.

WHEN TO USE IT over sqlite-vec:

  * A run that should leave nothing behind. sqlite-vec writes `ovat_index.db`
    into the working directory and it survives; this holds everything in the
    process and is gone when the process is.
  * A demo or a test that wants a real retriever without a file to clean up
    between runs.
  * Any machine where SQLite is the problem rather than the solution. RAG was
    silently broken on every SQLite older than 3.38 once already; this path
    does not involve SQLite at all.

WHEN NOT TO. Nothing persists, so `ovat index` followed by a separate
`ovat run` retrieves nothing: the index died with the indexing process. That
is not a bug to fix later, it is the trade being made, so `unavailable`-style
honesty applies -- the docstring says it and the README table says it.

THE SEARCH IS EXACT, not approximate. Every vector is compared, which is
O(n) per query and completely fine at the scale OVAT targets (a folder of
notes, not a web index). sqlite-vec is also brute force, so this is not a
downgrade; a true ANN backend (usearch, hnsw) would slot in beside both
without touching anything here.
"""
import threading

import numpy as np

from ovat.providers.base import EmbeddingsProvider, RetrieverProvider


class InMemoryRetrieverProvider(RetrieverProvider):
    """Vectors in a numpy array, texts in a list, nothing on disk."""

    def __init__(self, embedder: EmbeddingsProvider, dim: int = 384):
        self.embedder = embedder
        self.dim = dim
        self._texts: list[str] = []
        self._sources: list[str | None] = []
        # (n, dim) float32. None until the first add, because numpy needs a
        # width and an empty (0, dim) array is a needless special case in
        # every method below.
        self._vectors: np.ndarray | None = None
        # The same reason retriever_sqlitevec passes check_same_thread=False:
        # LangChain runs tools on a worker thread, so add() and retrieve() can
        # genuinely overlap. A list append is not atomic with respect to the
        # numpy array it must stay index-aligned with, and a retrieve() that
        # caught them out of step would return the wrong text for a vector --
        # a wrong citation, which is worse than an error.
        self._write_lock = threading.Lock()
        self._closed = False

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Drop everything held. Idempotent, like the sqlite-vec one.

        There is no connection to release, but closing must still mean the
        retriever stops answering: a caller that closes and then retrieves has
        a bug, and silently serving stale results hides it.
        """
        with self._write_lock:
            self._texts = []
            self._sources = []
            self._vectors = None
            self._closed = True

    def __enter__(self) -> "InMemoryRetrieverProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Retriever closed")

    # -- writing ------------------------------------------------------------

    def _forget_source(self, source: str) -> None:
        """Drop everything previously stored for one source.

        Kept index-aligned by rebuilding all three containers from the same
        keep-list. Deleting from the lists and the array separately is how the
        two drift, and a vector paired with the wrong text is a wrong citation
        that nothing downstream can detect.
        """
        keep = [i for i, s in enumerate(self._sources) if s != source]
        if len(keep) == len(self._sources):
            return
        self._texts = [self._texts[i] for i in keep]
        self._sources = [self._sources[i] for i in keep]
        self._vectors = (self._vectors[keep] if keep and self._vectors is not None
                         else None)

    def add(self, texts: list[str], sources: list[str] | None = None) -> None:
        """Embed each text and hold the vector + text + optional source.

        Adding a source REPLACES whatever that source had before, so indexing
        is idempotent and matches sqlite-vec exactly: re-running `ovat index`
        on an unchanged folder leaves the same index, and re-indexing an edited
        file drops the sentences it no longer contains.

        With no sources there is no key to replace on, so those chunks are
        appended. Same deliberate limit as the other backend.
        """
        self._check_open()
        if not texts:
            return
        # Outside the lock: embedding is pure compute and can be slow enough
        # to matter, and it touches nothing shared.
        vectors = np.asarray(self.embedder.embed(texts), dtype=np.float32)
        with self._write_lock:
            self._check_open()
            if sources:
                # dict.fromkeys keeps first-seen order and drops repeats; the
                # indexer passes one file's source repeated per chunk, but the
                # signature allows a mixed batch and this handles both.
                for source in dict.fromkeys(sources):
                    self._forget_source(source)
            self._texts.extend(texts)
            self._sources.extend(sources if sources else [None] * len(texts))
            self._vectors = (vectors if self._vectors is None
                             else np.vstack([self._vectors, vectors]))

    # -- reading ------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """The nearest `top_k` chunks, closest first.

        COSINE DISTANCE, reported as `distance` so that smaller is closer --
        the same direction sqlite-vec's L2 uses, because search_docs and the
        loop's citation code read this field without knowing which backend
        produced it. For the normalised embeddings a sentence-transformer
        emits, cosine and L2 rank identically, so the two backends agree on
        ordering rather than merely on format.
        """
        self._check_open()
        if self._vectors is None or not self._texts:
            return []
        query_vector = np.asarray(self.embedder.embed([query])[0],
                                  dtype=np.float32)

        # Normalise both sides, guarding the zero vector: an empty or degenerate
        # embedding would otherwise divide by zero and hand back nan, which
        # sorts unpredictably and would silently reorder results.
        def unit(a, axis=None):
            norm = np.linalg.norm(a, axis=axis, keepdims=axis is not None)
            return a / np.where(norm == 0, 1.0, norm)

        similarity = unit(self._vectors, axis=1) @ unit(query_vector)
        distances = 1.0 - similarity

        # argsort over the whole set: exact, and cheap at this scale. top_k is
        # clamped because asking for more than exists is a reasonable thing for
        # a caller to do and must not raise.
        order = np.argsort(distances)[:max(0, top_k)]
        return [{"text": self._texts[i],
                 "source": self._sources[i],
                 "distance": float(distances[i])}
                for i in order]
