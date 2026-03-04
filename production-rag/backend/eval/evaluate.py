"""Run the RAG pipeline over the golden dataset and score it.

Usage:
    python -m eval.evaluate                  # uses .env config + Anthropic API
    python -m eval.evaluate --limit 3        # quick smoke run on 3 questions

Requires a valid ANTHROPIC_API_KEY because the metrics use an LLM judge.
Writes eval_report.json and eval_report.md next to this file.
"""

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from app.config import get_settings
from app.embeddings import get_embedder
from app.main import build_state
from eval import metrics

HERE = Path(__file__).resolve().parent


def _load_dataset(limit: int | None) -> list[dict]:
    data = json.loads((HERE / "golden_dataset.json").read_text(encoding="utf-8"))["dataset"]
    return data[:limit] if limit else data


async def _score_row(row, pipeline, embedder, settings) -> dict:
    result = await pipeline.answer(row["question"])
    answer = result["answer"]
    contexts = [c["text"] for c in result["citations"]]
    context_blob = "\n\n".join(contexts)
    judge = settings.guardrail_model

    faith = await metrics.faithfulness(pipeline._llm, answer, context_blob, model=judge)
    relevancy = await metrics.answer_relevancy(
        pipeline._llm, embedder, row["question"], answer, model=judge
    )
    precision = await metrics.context_precision(
        pipeline._llm, row["question"], contexts, model=judge
    )
    similarity = metrics.answer_similarity(embedder, answer, row["reference"])

    return {
        "question": row["question"],
        "answer": answer,
        "refused": result["guardrails"]["refused"],
        "faithfulness": round(faith, 3),
        "answer_relevancy": round(relevancy, 3),
        "context_precision": round(precision, 3),
        "answer_similarity": round(similarity, 3),
    }


def _aggregate(rows: list[dict]) -> dict:
    keys = ["faithfulness", "answer_relevancy", "context_precision", "answer_similarity"]
    return {k: round(statistics.mean(r[k] for r in rows), 3) for k in keys}


def _write_markdown(rows: list[dict], summary: dict) -> str:
    lines = ["# RAG Evaluation Report", "", "## Summary (mean across dataset)", ""]
    lines += [f"- **{k}**: {v}" for k, v in summary.items()]
    lines += ["", "## Per-question", "", "| Question | Faith | Relevancy | Ctx Prec | Ans Sim |", "|---|---|---|---|---|"]
    for r in rows:
        q = r["question"][:60] + ("..." if len(r["question"]) > 60 else "")
        lines.append(
            f"| {q} | {r['faithfulness']} | {r['answer_relevancy']} | "
            f"{r['context_precision']} | {r['answer_similarity']} |"
        )
    return "\n".join(lines) + "\n"


async def main(limit: int | None) -> None:
    settings = get_settings()
    state = build_state(settings)
    pipeline = state["pipeline"]
    embedder = get_embedder(settings)

    if state["vectorstore"].count() == 0:
        from app.ingest import seed_sample_documents

        seed_sample_documents(embedder, state["vectorstore"], state["bm25"], settings)

    dataset = _load_dataset(limit)
    rows = []
    for i, row in enumerate(dataset, start=1):
        print(f"[{i}/{len(dataset)}] {row['question'][:70]}")
        rows.append(await _score_row(row, pipeline, embedder, settings))

    summary = _aggregate(rows)
    report = {"summary": summary, "results": rows}
    (HERE / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (HERE / "eval_report.md").write_text(_write_markdown(rows, summary), encoding="utf-8")

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key:>20}: {value}")
    print("\nWrote eval_report.json and eval_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="evaluate only N questions")
    args = parser.parse_args()
    asyncio.run(main(args.limit))
