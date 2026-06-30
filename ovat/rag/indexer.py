# ovat/rag/indexer.py
"""Read documents off disk, slice them into chunks, hand them to a retriever.

Why chunk at all: an embedding model squeezes a whole input into one fixed-size
vector. Feed it a 50-page file and the meaning blurs into mush. Feed it a
paragraph and the vector actually captures that paragraph. So I cut each file
into overlapping windows, embed those, and store them. At query time the search
returns the handful of windows closest in meaning, not the whole file.

The retriever argument is any RetrieverProvider. I never name a concrete class
here, so the same indexer works whether the vectors land in sqlite-vec today or
some other backend tomorrow.
"""
from pathlib import Path

from ovat.providers.base import RetrieverProvider

# The file types I treat as plain text. Keeping this small and explicit means I
# never try to embed a PDF or an image as if it were UTF-8 text.
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def chunk_text(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """Slice text into overlapping windows of `size` characters.

    The overlap matters: if I cut on an exact boundary I can split a sentence
    so neither half carries its full meaning. Sharing `overlap` characters with
    the next window keeps ideas that straddle a cut findable from both sides.

    I work in characters, not tokens, on purpose. It needs no tokenizer, it is
    deterministic, and for a demo-scale corpus the difference is not worth a
    heavier dependency.
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
    """Yield every readable text file under `folder`, recursively.

    I sort the paths so indexing the same folder twice processes files in the
    same order. Reproducible runs are easier to reason about and to test.
    """
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"No such folder to index: {folder}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def index_folder(folder: str, retriever: RetrieverProvider,
                 size: int = 512, overlap: int = 64) -> dict:
    """Index every text file under `folder` into `retriever`.

    Returns a small summary dict so the CLI can print something honest like
    "indexed 12 chunks from 3 files" instead of a silent success.

    The source I store is the file path. That is what lets search_docs answer
    "where did this come from", which is the citation half of RAG.
    """
    total_files = 0
    total_chunks = 0
    for path in iter_text_files(folder):
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, size=size, overlap=overlap)
        if not chunks:
            continue
        # One source string per chunk, all pointing back at this file.
        sources = [str(path)] * len(chunks)
        retriever.add(chunks, sources=sources)
        total_files += 1
        total_chunks += len(chunks)
    return {"files": total_files, "chunks": total_chunks}
