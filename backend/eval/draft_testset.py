"""Draft candidate question/ground-truth pairs from YOUR indexed documents.

The output is a starting point only: review every pair by hand, fix or delete
weak ones, then copy 15-20 good ones into eval/testset.json. LLM-written
ground truths that nobody checked will make RAGAS scores meaningless.

Usage (from backend/, after ingesting):
    uv run python -m eval.draft_testset --n 25
"""

import argparse
import json
import random
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import settings
from app.retrievers import all_chunks

EVAL_DIR = Path(__file__).resolve().parent

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write ONE question an analyst would realistically ask that is fully answered by "
            "the passage, and its concise ground-truth answer using only facts in the passage. "
            "Prefer specific facts (numbers, names, dates, codes, identifiers). The question "
            "must make sense without seeing the passage.",
        ),
        ("human", "Passage (from {source}):\n{text}"),
    ]
)


class QAPair(BaseModel):
    question: str
    ground_truth: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=25, help="number of candidates to draft")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "testset.draft.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chunks = [c for c in all_chunks() if len(c.page_content) > 200]
    if not chunks:
        raise SystemExit("No chunks found - run `uv run python -m app.ingest` first.")
    random.seed(args.seed)
    picked = random.sample(chunks, min(args.n, len(chunks)))

    llm = ChatOpenAI(model=settings.llm_model, temperature=0.3, api_key=settings.openai_api_key)
    chain = PROMPT | llm.with_structured_output(QAPair)
    pairs = chain.batch(
        [{"source": c.metadata.get("source", "unknown"), "text": c.page_content} for c in picked],
        config={"max_concurrency": 4},
    )
    drafts = [
        {**p.model_dump(), "_source": c.metadata.get("source")} for p, c in zip(pairs, picked)
    ]
    args.out.write_text(json.dumps(drafts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(drafts)} drafts to {args.out} - review them before use.")


if __name__ == "__main__":
    main()
