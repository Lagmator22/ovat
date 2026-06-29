# ovat/rag/__init__.py
"""RAG building blocks: turn a folder of documents into a searchable index.

This package is deliberately small. The heavy lifting (vectors, search) lives in
the providers behind their ABCs. What is here is the glue: slice files into
chunks and feed them to whatever RetrieverProvider the YAML selected.
"""
