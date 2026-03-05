"""Optional: score the pipeline with the RAGAS library (instead of eval/metrics.py).

This is provided so results can be cross-checked against the standard tool. It is
intentionally kept out of the serving image; install its extra deps first:

    pip install -r eval/requirements-eval.txt
    python -m eval.run_ragas --limit 3

RAGAS is pinned in requirements-eval.txt because its API changes between minor
versions. Requires ANTHROPIC_API_KEY and network access (downloads the embedding
model the first time when EMBEDDING_BACKEND=fastembed).
"""

import argparse
import asyncio
import json
from pathlib import Path

from app.config import get_settings
from app.embeddings import get_embedder
from app.main import build_state

HERE = Path(__file__).resolve().parent


async def _collect_samples(pipeline, dataset):
    samples = []
    for row in dataset:
        result = await pipeline.answer(row["question"])
        samples.append(
            {
                "user_input": row["question"],
                "response": result["answer"],
                "retrieved_contexts": [c["text"] for c in result["citations"]] or [""],
                "reference": row["reference"],
            }
        )
    return samples


def main(limit: int | None) -> None:
    from langchain_groq import ChatGroq
    from langchain_core.embeddings import Embeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        ResponseRelevancy,
    )

    settings = get_settings()
    state = build_state(settings)
    pipeline = state["pipeline"]
    embedder = get_embedder(settings)

    if state["vectorstore"].count() == 0:
        from app.ingest import seed_sample_documents

        seed_sample_documents(embedder, state["vectorstore"], state["bm25"], settings)

    data = json.loads((HERE / "golden_dataset.json").read_text(encoding="utf-8"))["dataset"]
    if limit:
        data = data[:limit]

    samples = asyncio.run(_collect_samples(pipeline, data))
    eval_dataset = EvaluationDataset.from_list(samples)

    judge = LangchainLLMWrapper(ChatGroq(model=settings.guardrail_model, temperature=0))

    # A LangChain Embeddings subclass wrapping our project embedder.
    class _Embeddings(Embeddings):
        def embed_query(self, text: str) -> list[float]:
            return embedder.embed_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return embedder.embed_documents(texts)

    embeddings = LangchainEmbeddingsWrapper(_Embeddings())

    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
        ],
        llm=judge,
        embeddings=embeddings,
    )
    print(result)
    result.to_pandas().to_csv(HERE / "ragas_report.csv", index=False)
    print("Wrote ragas_report.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.limit)
