from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import RetrieverMode, settings
from app.retrievers import get_retriever

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research analyst. Answer the question using ONLY the numbered "
            "context passages. Cite passages inline like [1]. If the context does not "
            "contain the answer, say you don't know - do not guess.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


@dataclass
class RagResult:
    answer: str
    contexts: list[Document]
    retriever: RetrieverMode


@lru_cache
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.openai_api_key)


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {d.metadata.get('source', 'unknown')})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


async def answer_question(question: str, mode: RetrieverMode | None = None) -> RagResult:
    mode = mode or settings.retriever_mode
    docs = await get_retriever(mode).ainvoke(question)
    chain = PROMPT | get_llm() | StrOutputParser()
    answer = await chain.ainvoke({"context": format_context(docs), "question": question})
    return RagResult(answer=answer, contexts=docs, retriever=mode)
