# ovat/providers/retriever_sqlitevec.py
"""Layer 4: concrete Retriever plug: store vectors, find the closest ones.

Fills the RetrieverProvider socket using sqlite-vec (a SQLite extension that
adds vector search). This is same as OvaSearch's USearch logic, but re-done with sqlite-vec.

Design note: this plug HAS-A embedder (composition). It can't turn text into
vectors itself, so we hand it an EmbeddingsProvider. Same idea as a C++ class
that holds a pointer to a helper object it delegates to. We use composition here.
"""
import sqlite3

import sqlite_vec

from ovat.providers.base import RetrieverProvider, EmbeddingsProvider


class SQLiteVecRetrieverProvider(RetrieverProvider):
    """Vector storage + nearest-neighbour search via sqlite-vec."""

    def __init__(self, embedder: EmbeddingsProvider, dim: int = 384,
                 db_path: str = ":memory:"):
        self.embedder = embedder          # the helper that makes vectors
        self.dim = dim                    # bge-small -> 384 numbers per vector
        # Open a SQLite database and load the vector-search extension into it.
        self.db = sqlite3.connect(db_path)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        # A virtual table that stores 384-float vectors and can search them.
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS docs USING vec0(embedding float[{self.dim}])"
        )
        self.texts: list[str] = []        # row i of the table -> texts[i]

    def add(self, texts: list[str]) -> None:
        vectors = self.embedder.embed(texts)          # text -> vectors
        for text, vec in zip(texts, vectors):
            rowid = len(self.texts)
            self.texts.append(text)
            self.db.execute(
                "INSERT INTO docs(rowid, embedding) VALUES (?, ?)",
                (rowid, sqlite_vec.serialize_float32(vec)),  # pack floats -> bytes
            )
        self.db.commit()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        qvec = self.embedder.embed([query])[0]         # embed the question
        rows = self.db.execute(
            "SELECT rowid, distance FROM docs "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (sqlite_vec.serialize_float32(qvec), top_k),
        ).fetchall()
        # Smaller distance = closer in meaning = better match [semantic search basically]
        return [{"text": self.texts[r[0]], "distance": r[1]} for r in rows]
