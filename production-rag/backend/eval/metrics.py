"""RAG quality metrics, implemented from first principles.

These mirror the RAGAS definitions so the harness has no version-fragile heavy
dependency. A drop-in variant using the `ragas` library lives in run_ragas.py.

Metrics:
- faithfulness        : fraction of answer claims supported by retrieved context.
- answer_relevancy    : how well the answer addresses the question (RAGAS approach:
                        generate questions the answer would answer, then measure
                        embedding similarity to the original question).
- context_precision   : fraction of retrieved chunks judged relevant to the question.
- answer_similarity   : embedding similarity between the answer and the reference.
"""

import json
import math

from app.guardrails import check_faithfulness

_RELEVANCY_SYSTEM = (
    "Given an ANSWER, generate exactly 3 distinct questions that this answer would "
    "directly and fully address. Respond with ONLY a JSON array of 3 strings."
)

_CONTEXT_SYSTEM = (
    "Decide whether the CONTEXT passage is relevant and useful for answering the "
    "QUESTION. Respond with ONLY a JSON object: {\"relevant\": true} or "
    "{\"relevant\": false}."
)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _parse_json(raw: str):
    raw = raw.strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


async def faithfulness(llm, answer: str, context: str, *, model: str) -> float:
    report = await check_faithfulness(llm, answer, context, model=model)
    return report.faithfulness_score


async def answer_relevancy(llm, embedder, question: str, answer: str, *, model: str) -> float:
    raw = await llm.complete(
        system=_RELEVANCY_SYSTEM, user=f"ANSWER:\n{answer}", model=model, max_tokens=300
    )
    generated = _parse_json(raw)
    if not isinstance(generated, list) or not generated:
        return 0.0
    q_vec = embedder.embed_query(question)
    sims = [_cosine(q_vec, embedder.embed_query(str(g))) for g in generated]
    return max(0.0, sum(sims) / len(sims))


async def context_precision(llm, question: str, contexts: list[str], *, model: str) -> float:
    if not contexts:
        return 0.0
    relevant = 0
    for ctx in contexts:
        raw = await llm.complete(
            system=_CONTEXT_SYSTEM,
            user=f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}",
            model=model,
            max_tokens=50,
        )
        parsed = _parse_json(raw)
        if isinstance(parsed, dict) and parsed.get("relevant"):
            relevant += 1
    return relevant / len(contexts)


def answer_similarity(embedder, answer: str, reference: str) -> float:
    return max(0.0, _cosine(embedder.embed_query(answer), embedder.embed_query(reference)))
