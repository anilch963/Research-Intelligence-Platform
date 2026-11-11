from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import RetrieverMode, settings
from app.rag import answer_question
from app.retrievers import chunk_count

app = FastAPI(title="Research Intelligence API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    retriever: RetrieverMode | None = None


class Source(BaseModel):
    source: str
    page: int | None = None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    retriever: RetrieverMode
    sources: list[Source]


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "chunks_indexed": chunk_count(),
        "retriever_mode": settings.retriever_mode,
        "llm_model": settings.llm_model,
    }


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    if chunk_count() == 0:
        raise HTTPException(503, "Index is empty - run `uv run python -m app.ingest` first.")
    result = await answer_question(req.question, req.retriever)
    return QueryResponse(
        answer=result.answer,
        retriever=result.retriever,
        sources=[
            Source(
                source=d.metadata.get("source", "unknown"),
                page=d.metadata.get("page"),
                snippet=d.page_content[:300],
            )
            for d in result.contexts
        ],
    )
