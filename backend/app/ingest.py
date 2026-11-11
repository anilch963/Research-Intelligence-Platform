"""Load documents from data/docs, chunk them, and (re)build the Chroma collection.

Usage (from backend/):
    uv run python -m app.ingest            # rebuild from scratch
    uv run python -m app.ingest --append   # add to existing collection
"""

import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.retrievers import get_vectorstore, reset_caches

TEXT_SUFFIXES = {".md", ".txt"}


def load_documents(docs_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() in TEXT_SUFFIXES:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    for doc in docs:
        doc.metadata["source"] = Path(doc.metadata["source"]).name
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def build_index(reset: bool = True) -> int:
    docs = load_documents(settings.docs_dir)
    if not docs:
        raise SystemExit(f"No .pdf/.md/.txt files found in {settings.docs_dir}")
    chunks = split_documents(docs)

    vectorstore = get_vectorstore()
    if reset:
        vectorstore.reset_collection()
    ids = [
        f"{c.metadata['source']}:{c.metadata.get('page', 0)}:{c.metadata['start_index']}"
        for c in chunks
    ]
    vectorstore.add_documents(chunks, ids=ids)
    reset_caches()
    print(f"Indexed {len(docs)} documents -> {len(chunks)} chunks into '{settings.collection_name}'")
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--append", action="store_true", help="don't wipe the collection first")
    build_index(reset=not parser.parse_args().append)
