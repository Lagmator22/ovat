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
import sys
import threading

import sqlite_vec

from ovat.providers.base import RetrieverProvider, EmbeddingsProvider


class SQLiteVecRetrieverProvider(RetrieverProvider):
    """Vector storage + nearest-neighbour search via sqlite-vec."""

    def __init__(self, embedder: EmbeddingsProvider, dim: int = 384,
                 db_path: str = ":memory:"):
        self.embedder = embedder          # the helper that makes vectors
        # check_same_thread=False below lets worker threads use this connection,
        # which SQLite allows for READS but not for concurrent WRITES: two
        # simultaneous add() calls would raise "database is locked". Today the
        # only writer is `ovat index` (single-threaded) while the LangChain
        # worker thread only retrieves, so this is insurance rather than a
        # live bug -- but it is one line, and the alternative failure is a
        # crash mid-index.
        self._write_lock = threading.Lock()
        self.dim = dim                    # bge-small -> 384 numbers per vector
        # Open a SQLite database and load the vector-search extension into it.
        # check_same_thread=False because the LangChain (react) engine runs tool
        # calls on a worker thread, not the thread that built the retriever. The
        # agent loop is sequential (one query at a time), so sharing the single
        # connection across threads is safe here; SQLite just blocks it by default.
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        # sqlite-vec is a loadable SQLite EXTENSION, and a Python can be built
        # without the ability to load one: CPython compiled with
        # --disable-loadable-sqlite-extensions simply has no
        # enable_load_extension method. Homebrew and python.org macOS builds
        # ship that way, and so does the macOS Python on GitHub's runners.
        #
        # Unguarded, the first thing a user saw was
        #   AttributeError: 'sqlite3.Connection' object has no attribute
        #   'enable_load_extension'
        # which says nothing about SQLite extensions, nothing about their
        # Python, and nothing about what to do. Rule 6: a failure a person has
        # to act on is a sentence, not a traceback.
        if not hasattr(self.db, "enable_load_extension"):
            self.db.close()
            raise RuntimeError(
                f"This Python cannot load SQLite extensions, and vector "
                f"search needs one (sqlite-vec).\n"
                f"  interpreter: {sys.executable}\n"
                f"It was built with --disable-loadable-sqlite-extensions, "
                f"which is common for Homebrew and python.org macOS builds.\n"
                f"Fix it one of these ways:\n"
                f"  1. use a Python that allows extensions "
                f"(pyenv, uv, conda, or your distro's python3)\n"
                f"  2. on macOS: brew install python and use that interpreter\n"
                f"  3. leave the rag: block out of your workflow; search_docs "
                f"then answers in stub mode and everything else still works")
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

    def close(self) -> None:
        """Close the SQLite connection and flush everything to disk.

        Note to myself: __init__ acquires a real OS resource (the connection),
        so something must release it; Python has no destructor I can rely on
        the way C++ does. Safe to call twice: closing an already-closed
        connection is a no-op here because I null the handle after the first.
        """
        if self.db is not None:
            self.db.close()
            self.db = None

    # `with SQLiteVecRetrieverProvider(...) as r:` is Python's RAII. __exit__
    # runs on ANY exit from the block, exception or not, like a destructor.
    def __enter__(self) -> "SQLiteVecRetrieverProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _next_rowid(self) -> int:
        # Derive the next id from the table on disk, so a reopened database does
        # not reuse an id that already exists (which used to crash add()).
        row = self.db.execute("SELECT COALESCE(MAX(rowid), -1) + 1 FROM chunks").fetchone()
        return int(row[0])

    def _check_open(self) -> None:
        if self.db is None:
            raise RuntimeError("Retriever closed")

    def _forget_source(self, source: str) -> None:
        """Remove everything previously stored for one source.

        Two deletes, not one, because the two tables are keyed the same way but
        shaped differently: `chunks` HAS a source column, the vec0 virtual
        table `docs` does not. So the rowids have to be looked up in chunks
        first and then deleted from docs by rowid.

        Getting this half-right is worse than not doing it: a vector whose
        chunk is gone is an orphan, retrieve() matches it, finds no chunk, and
        silently skips it. You would ask for top_k=5 and quietly get 2 results
        with no error anywhere.
        """
        self._check_open()
        rowids = [row[0] for row in self.db.execute(
            "SELECT rowid FROM chunks WHERE source = ?", (source,))]
        if rowids:
            self.db.executemany("DELETE FROM docs WHERE rowid = ?", [(r,) for r in rowids])
        self.db.execute("DELETE FROM chunks WHERE source = ?", (source,))

    def add(self, texts: list[str], sources: list[str] | None = None) -> None:
        """Embed each text and store the vector + text + optional source.

        Note: sources is optional and lines up with texts by index. It lets the
        agent answer "with source citations" later.

        Adding a source REPLACES whatever that source had before, so indexing
        is idempotent: running `ovat index` twice on an unchanged folder leaves
        the same index, and re-indexing an edited file drops the sentences it
        no longer contains. It used to append, so three runs put the same chunk
        in three times and retrieve(top_k=3) returned it three times, crowding
        out every other document.

        With no sources there is no key to replace ON, so those chunks are
        appended. That is a deliberate limit of the contract, not an oversight.
        """
        self._check_open()
        vectors = self.embedder.embed(texts)   # outside the lock: pure compute
        with self._write_lock:
            self._add_locked(texts, vectors, sources)

    def _add_locked(self, texts: list[str], vectors: list,
                    sources: list[str] | None) -> None:
        """The write half of add(), serialised by the caller's lock."""
        if sources:
            # dict.fromkeys keeps first-seen order and drops repeats; the
            # indexer passes one file's source repeated per chunk, but the
            # signature allows a mixed batch and this handles both.
            for source in dict.fromkeys(sources):
                self._forget_source(source)
        # Delete and insert share one transaction (the commit below), so a
        # crash midway cannot leave the index emptier than it started.
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
        self._check_open()
        qvec = self.embedder.embed([query])[0]         # embed the question
        # Step 1: nearest-neighbour search on the vector table.
        # `k = ?`, not `LIMIT ?`. Both express "give me the nearest top_k", but
        # LIMIT only reaches a virtual table if SQLite pushes it down into
        # xBestIndex, and SQLite only started doing that in 3.38. On anything
        # older sqlite-vec never sees a bound and refuses the query outright:
        #
        #   sqlite3.OperationalError: A LIMIT or 'k = ?' constraint is
        #   required on vec0 knn queries.
        #
        # Ubuntu 22.04 -- a platform this project's own README supports --
        # ships SQLite 3.37.2. Windows ships 3.50.4 and Ubuntu 24.04 ships
        # 3.45, so the same sqlite-vec 0.1.9 worked on every machine this had
        # ever been run on, and RAG was silently broken for everyone else:
        # `ovat index` reported success, the query returned nothing, and the
        # model improvised an answer with no citation.
        #
        # `k = ?` is sqlite-vec's own KNN form and needs no push-down, so it
        # behaves identically on both sides of 3.38.
        rows = self.db.execute(
            "SELECT rowid, distance FROM docs "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
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
