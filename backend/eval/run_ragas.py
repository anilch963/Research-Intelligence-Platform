"""Run the RAG pipeline over the test set and score it with RAGAS.

Usage (from backend/):
    uv run python -m eval.run_ragas --mode vector --out eval/results/baseline.json
    uv run python -m eval.run_ragas --mode hybrid --out eval/results/hybrid.json
"""

import argparse
import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, ResponseRelevancy

from app.config import settings
from app.rag import answer_question
from app.retrievers import chunk_count, get_embeddings

EVAL_DIR = Path(__file__).resolve().parent


async def generate_samples(testset: list[dict], mode: str, concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(item: dict) -> dict:
        async with semaphore:
            result = await answer_question(item["question"], mode)
        return {
            "user_input": item["question"],
            "retrieved_contexts": [d.page_content for d in result.contexts],
            "response": result.answer,
            "reference": item["ground_truth"],
        }

    return await asyncio.gather(*(run_one(item) for item in testset))


def score(samples: list[dict]) -> tuple[dict, list[dict]]:
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference()]
    result = evaluate(
        EvaluationDataset.from_list(samples),
        metrics=metrics,
        llm=LangchainLLMWrapper(
            ChatOpenAI(model=settings.eval_llm_model, temperature=0, api_key=settings.openai_api_key)
        ),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
        run_config=RunConfig(max_workers=8, timeout=180),
    )
    df = result.to_pandas()
    names = [m.name for m in metrics]
    summary = {name: round(float(df[name].mean()), 4) for name in names}
    rows = [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["vector", "hybrid"], required=True)
    parser.add_argument("--testset", type=Path, default=EVAL_DIR / "testset.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    testset = json.loads(args.testset.read_text(encoding="utf-8"))
    if not 15 <= len(testset) <= 20:
        print(f"warning: test set has {len(testset)} questions (plan calls for 15-20)")
    if chunk_count() == 0:
        raise SystemExit("Index is empty - run `uv run python -m app.ingest` first.")

    print(f"Answering {len(testset)} questions with retriever={args.mode} ...")
    samples = asyncio.run(generate_samples(testset, args.mode, args.concurrency))
    print("Scoring with RAGAS ...")
    summary, rows = score(samples)

    report = {
        "retriever": args.mode,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "llm_model": settings.llm_model,
            "eval_llm_model": settings.eval_llm_model,
            "embedding_model": settings.embedding_model,
            "top_k": settings.top_k,
            "bm25_weight": settings.bm25_weight if args.mode == "hybrid" else None,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "num_questions": len(testset),
        },
        "summary": summary,
        "per_question": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResults ({args.mode}) -> {args.out}")
    for name, value in summary.items():
        print(f"  {name:<40} {value:.4f}")


if __name__ == "__main__":
    main()
