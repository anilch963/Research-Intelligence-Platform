"""Compare two RAGAS result files and optionally write the table into the README.

Usage (from backend/):
    uv run python -m eval.compare eval/results/baseline.json eval/results/hybrid.json
    uv run python -m eval.compare eval/results/baseline.json eval/results/hybrid.json --update-readme ../README.md
"""

import argparse
import json
import re
from pathlib import Path

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "llm_context_precision_with_reference": "Context precision",
}
README_START = "<!-- RAGAS_RESULTS_START -->"
README_END = "<!-- RAGAS_RESULTS_END -->"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_table(base: dict, new: dict) -> str:
    lines = [
        f"| Metric | Baseline ({base['retriever']}) | {new['retriever'].capitalize()} | Δ |",
        "|---|---:|---:|---:|",
    ]
    for key, label in METRIC_LABELS.items():
        b, n = base["summary"][key], new["summary"][key]
        lines.append(f"| {label} | {b:.3f} | {n:.3f} | {n - b:+.3f} |")
    cfg = new["config"]
    lines.append("")
    lines.append(
        f"_{cfg['num_questions']} questions · top_k={cfg['top_k']} · "
        f"bm25_weight={cfg['bm25_weight']} · generator={cfg['llm_model']} · "
        f"judge={cfg['eval_llm_model']} · run {new['timestamp'][:10]}_"
    )
    return "\n".join(lines)


def movers(base: dict, new: dict, metric: str, limit: int = 5) -> str:
    base_by_q = {r["user_input"]: r.get(metric) for r in base["per_question"]}
    deltas = []
    for row in new["per_question"]:
        b, n = base_by_q.get(row["user_input"]), row.get(metric)
        if b is not None and n is not None and n != b:
            deltas.append((n - b, row["user_input"]))
    if not deltas:
        return f"No per-question changes in {metric}."
    deltas.sort(reverse=True)
    gains = [d for d in deltas if d[0] > 0][:limit]
    drops = [d for d in reversed(deltas) if d[0] < 0][:limit]
    out = [f"Biggest per-question changes in {metric}:"]
    out += [f"  {delta:+.2f}  {question}" for delta, question in gains + drops]
    return "\n".join(out)


def update_readme(readme: Path, table: str) -> None:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + ".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"Markers {README_START} / {README_END} not found in {readme}")
    readme.write_text(pattern.sub(lambda _: f"{README_START}\n{table}\n{README_END}", text), encoding="utf-8")
    print(f"\nUpdated {readme}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--update-readme", type=Path, metavar="README")
    args = parser.parse_args()

    base, new = load(args.baseline), load(args.candidate)
    if base["config"]["num_questions"] != new["config"]["num_questions"]:
        print("warning: runs used different numbers of questions - comparison is not like-for-like")

    table = summary_table(base, new)
    print(table)
    print()
    print(movers(base, new, "llm_context_precision_with_reference"))
    if args.update_readme:
        update_readme(args.update_readme, table)


if __name__ == "__main__":
    main()
