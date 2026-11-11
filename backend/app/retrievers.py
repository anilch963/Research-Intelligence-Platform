"""Vector, BM25 and hybrid (EnsembleRetriever) retrievers over one shared chunk set.

BM25 is built from the chunks already stored in Chroma, so both retrievers
always search exactly the same corpus.
"""

import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import OpenAIEmbeddings

try:  # LangChain >= 1.0 moved legacy retrievers into langchain-classic
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    from langchain.retrievers import EnsembleRetriever

from app.config import RetrieverMode, settings


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


@lru_cache
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def chunk_count() -> int:
    return len(get_vectorstore().get(include=[])["ids"])


def all_chunks() -> list[Document]:
    data = get_vectorstore().get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]


def _tokenize(text: str) -> list[str]:
    # BM25Retriever's default is str.split(), which is case-sensitive and keeps
    # punctuation ("TL-1042." != "tl-1042"). Lowercased word tokens match far better.
    return re.findall(r"\w+", text.lower())


def vector_retriever(k: int) -> Runnable:
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


@lru_cache
def bm25_retriever(k: int) -> BM25Retriever:
    chunks = all_chunks()
    if not chunks:
        raise RuntimeError("Chroma collection is empty - run `python -m app.ingest` first.")
    retriever = BM25Retriever.from_documents(chunks, preprocess_func=_tokenize)
    retriever.k = k
    return retriever


def hybrid_retriever(k: int, bm25_weight: float) -> Runnable:
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever(k), vector_retriever(k)],
        weights=[bm25_weight, 1 - bm25_weight],
    )
    # EnsembleRetriever returns the RRF-ranked union of both lists (up to 2k docs).
    # Trim to k so hybrid gets the same context budget as vector-only; otherwise
    # the before/after RAGAS comparison is not apples-to-apples.
    return ensemble | RunnableLambda(lambda docs: docs[:k])


def get_retriever(mode: RetrieverMode | None = None, k: int | None = None) -> Runnable:
    mode = mode or settings.retriever_mode
    k = k or settings.top_k
    if mode == "vector":
        return vector_retriever(k)
    if mode == "hybrid":
        return hybrid_retriever(k, settings.bm25_weight)
    raise ValueError(f"Unknown retriever mode: {mode}")


def reset_caches() -> None:
    """Call after re-ingesting so BM25 is rebuilt from the new chunks."""
    bm25_retriever.cache_clear()
    get_vectorstore.cache_clear()
