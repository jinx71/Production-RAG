# Evaluation Harness

Two ways to evaluate retrieval + generation quality offline against
`golden_dataset.json` (10 Q&A pairs grounded in the sample SOPs).

## 1. Built-in metrics (default, no extra deps)

```bash
python -m eval.evaluate            # full dataset
python -m eval.evaluate --limit 3  # quick run
```

Implemented in `metrics.py` from first principles, mirroring the RAGAS definitions:

| Metric | What it measures |
|---|---|
| `faithfulness` | fraction of answer claims supported by retrieved context |
| `answer_relevancy` | how well the answer addresses the question |
| `context_precision` | fraction of retrieved chunks relevant to the question |
| `answer_similarity` | embedding similarity between answer and reference |

Outputs `eval_report.json` and `eval_report.md`.

## 2. RAGAS library (optional cross-check)

```bash
pip install -r eval/requirements-eval.txt
python -m eval.run_ragas --limit 3
```

Uses the real `ragas` package (`Faithfulness`, `ResponseRelevancy`,
`LLMContextPrecisionWithReference`) with Claude as the judge. RAGAS is pinned
because its API shifts between minor versions.

Both paths need a valid `ANTHROPIC_API_KEY` (the metrics use an LLM judge) and,
with `EMBEDDING_BACKEND=fastembed`, download the embedding model on first run.
