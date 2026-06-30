# ovat/providers/retriever_sqlitevec.py
"""Layer 4: concrete Retriever plug: store vectors, find the closest ones.

Fills the RetrieverProvider socket using sqlite-vec (a SQLite extension that
adds vector search). This is the same idea as OvaSearch's USearch logic, redone
with sqlite-vec.

Design note: this plug HAS-A embedder (composition). It can't turn text into
vectors itself, so we hand it an EmbeddingsProvider. Same idea as a C++ class
that holds a pointer to a helper object it delegates to.

Persistence note: the text chunks (and their source) live in a real SQLite
table `chunks`, NOT in a Python list. That is what makes a file database
survive a restart: both the vectors and the text come back from disk, and new
rowids are derived from the table (MAX(rowid)+1), so there is no collision.
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
        # check_same_thread=False because the LangChain (react) engine runs tool
        # calls on a worker thread, not the thread that built the retriever. The
        # agent loop is sequential (one query at a time), so sharing the single
        # connection across threads is safe here; SQLite just blocks it by default.
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        # A virtual table that stores the vectors and can search them by distance.
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS docs USING vec0(embedding float[{self.dim}])"
        )
        # A normal table that persists the text + its source, keyed by the same
        # rowid as the vector. This is what fixes persistence across restarts.
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks "
            "(rowid INTEGER PRIMARY KEY, text TEXT NOT NULL, source TEXT)"
        )
        self.db.commit()

    def _next_rowid(self) -> int:
        # Derive the next id from the table on disk, so a reopened database does
        # not reuse an id that already exists (which used to crash add()).
        row = self.db.execute("SELECT COALESCE(MAX(rowid), -1) + 1 FROM chunks").fetchone()
        return int(row[0])

    def add(self, texts: list[str], sources: list[str] | None = None) -> None:
        """Embed each text and store the vector + text + optional source.

        Note: sources is optional and lines up with texts by index. It lets the
        agent answer "with source citations" later.
        """
        vectors = self.embedder.embed(texts)
        for i, (text, vec) in enumerate(zip(texts, vectors)):
            rowid = self._next_rowid()
            source = sources[i] if sources else None
            self.db.execute(
                "INSERT INTO chunks(rowid, text, source) VALUES (?, ?, ?)",
                (rowid, text, source),
            )
            self.db.execute(
                "INSERT INTO docs(rowid, embedding) VALUES (?, ?)",
                (rowid, sqlite_vec.serialize_float32(vec)),
            )
        self.db.commit()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        qvec = self.embedder.embed([query])[0]         # embed the question
        # Step 1: nearest-neighbour search on the vector table.
        rows = self.db.execute(
            "SELECT rowid, distance FROM docs "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (sqlite_vec.serialize_float32(qvec), top_k),
        ).fetchall()
        # Step 2: pull the matching text + source from the chunks table by rowid.
        # Smaller distance = closer in meaning = better match.
        results = []
        for rowid, distance in rows:
            chunk = self.db.execute(
                "SELECT text, source FROM chunks WHERE rowid = ?", (rowid,)
            ).fetchone()
            if chunk is not None:
                results.append({"text": chunk[0], "source": chunk[1], "distance": distance})
        return results
