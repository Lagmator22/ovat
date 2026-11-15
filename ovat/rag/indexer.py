# ovat/rag/indexer.py
"""Read text documents from disk, split them into overlapping chunks, and store
them in a ``RetrieverProvider``.

Documents are sliced into overlapping character windows so dense embeddings
preserve local context across chunk boundaries without requiring a tokenizer.
Depends only on the ``RetrieverProvider`` contract.
"""
import os
from pathlib import Path

from ovat.providers.base import RetrieverProvider

# Plain-text extensions indexed; binary formats (PDFs, images) are skipped.
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def chunk_text(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping character windows of ``size`` characters.

    Sharing ``overlap`` characters across consecutive windows preserves context
    around sentence boundaries without requiring a tokenizer dependency.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        # overlap >= size would mean the window never advances -> infinite loop.
        raise ValueError("overlap must be >= 0 and smaller than size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    step = size - overlap          # how far the window slides each time
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def iter_text_files(folder: str):
    """Yield every readable text file under ``folder`` in deterministic sorted order."""
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"No such folder to index: {folder}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def index_folder(folder: str, retriever: RetrieverProvider,
                 size: int = 512, overlap: int = 64,
                 on_progress=None) -> dict:
    """Index every text file under ``folder`` into ``retriever``.

    Returns a summary dictionary of processed files and stored chunks, with each
    chunk carrying its source file path for citations. ``on_progress(done,
    total, path)`` fires once per file after its embeddings have been persisted;
    the file list is materialised up front so callers can render accurate
    progress denominators.
    """
    paths = list(iter_text_files(folder))
    total = len(paths)
    total_files = 0
    total_chunks = 0
    indexed = {}
    for done, path in enumerate(paths, start=1):
        try:
            # utf-8-SIG, not utf-8: a file saved with a BOM (the Windows default
            # for several editors, and what Notepad writes) starts with U+FEFF,
            # and plain utf-8 keeps it. It then rode into the chunk, the vector
            # and the citation -- retrieval on an AI PC returned
            # '\ufeffThe Q3 review concluded...' as the stored text. utf-8-sig
            # strips it when present and is identical to utf-8 when it is not.
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
        except (OSError, UnicodeDecodeError):
            if on_progress is not None:
                on_progress(done, total, str(path))
            continue
        chunks = chunk_text(text, size=size, overlap=overlap)
        if chunks:
            # NORMALISED, because the source string is the key add() deletes
            # by before re-inserting. "docs\\notes.md" and ".\\docs\\notes.md"
            # are the same file and different strings, so indexing the same
            # folder twice via different path spellings appended instead of
            # replacing: one AI PC index held three identical copies of the
            # same document, which then crowd out every other result.
            key = os.path.normpath(str(path))
            sources = [key] * len(chunks)
            retriever.add(chunks, sources=sources)
            total_files += 1
            total_chunks += len(chunks)
            indexed[key] = {"path": os.path.abspath(path),
                            "mtime": os.path.getmtime(path)}
        if on_progress is not None:
            # Empty files are reported too. They cost nothing to skip, but a
            # bar that silently stops short of its total reads as a failure.
            on_progress(done, total, str(path))
    return {"files": total_files, "chunks": total_chunks, "indexed": indexed}


# Which files an index was built from, and when they were last changed. It
# lives next to the database: the vector store holds chunks, not file dates.
#
# Why it exists: on the AI PC the NPU example answered that the NPU "is
# explicitly not used for tool-calling agents", a claim the docs had dropped
# weeks earlier. Its document_qa.db was built before the correction, and
# nothing said the answer came from an old copy of the text.
MANIFEST_SUFFIX = ".sources.json"


from ovat.agentic.atomic import write_text


def record_index(db_path: str, indexed: dict) -> None:
    """Merge this run's {source: {path, mtime}} into the index's manifest."""
    if not db_path or db_path == ":memory:" or not indexed:
        return
    import json
    manifest = _read_manifest(db_path)
    manifest.update(indexed)
    write_text(db_path + MANIFEST_SUFFIX,
               json.dumps(manifest, indent=1, sort_keys=True))


def stale_sources(db_path: str) -> list[str]:
    """Indexed files that changed or vanished since they were indexed.

    An index with no manifest (built before OVAT recorded one) reports
    nothing: there is no date to compare against, and a warning on every run
    for every old index would teach people to ignore it.
    """
    stale = []
    for source, entry in sorted(_read_manifest(db_path).items()):
        try:
            if os.path.getmtime(entry["path"]) > entry["mtime"] + 1:
                stale.append(source)
        except (OSError, KeyError, TypeError):
            stale.append(source)          # deleted, or an unreadable entry
    return stale


def _read_manifest(db_path: str) -> dict:
    import json
    try:
        with open(db_path + MANIFEST_SUFFIX, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
