"""Hallucination guardrails.

Two layers of defence:
1. relevance_gate  -> cheap, deterministic pre-generation filter. If retrieval is
   weak (top dense similarity below threshold) we refuse without paying for a
   generation call, which blocks answers to out-of-domain questions.
2. check_faithfulness -> post-generation verifier. A cheap LLM judges whether the
   answer's claims are supported by the retrieved context, catching hallucinations
   that slip past the model's own instructions.
"""

import json
from dataclasses import dataclass

from app.generation import REFUSAL

_FAITH_SYSTEM = (
    "You are a strict fact-checker. Given an ANSWER and the CONTEXT it was supposedly "
    "based on, decide whether every factual claim in the answer is supported by the "
    "context.\n"
    "Respond with ONLY a JSON object, no prose, of the form:\n"
    '{"grounded": true, "faithfulness_score": 0.0, "unsupported_claims": ["..."]}\n'
    "faithfulness_score is the fraction of claims supported by the context (0.0-1.0)."
)


@dataclass
class FaithfulnessReport:
    grounded: bool
    faithfulness_score: float
    unsupported_claims: list[str]
    checked: bool = True


def relevance_gate(top_dense_score: float, has_results: bool, threshold: float) -> bool:
    return has_results and top_dense_score >= threshold


def _safe_parse(raw: str) -> dict | None:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


async def check_faithfulness(llm, answer: str, context: str, *, model: str) -> FaithfulnessReport:
    # A refusal is trivially grounded; skip the (paid) verification call.
    if answer.strip() == REFUSAL:
        return FaithfulnessReport(True, 1.0, [])

    user = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    try:
        raw = await llm.complete(
            system=_FAITH_SYSTEM, user=user, model=model, max_tokens=400, temperature=0.0
        )
    except Exception:
        # Fail open: never block answering because the verifier itself errored.
        return FaithfulnessReport(True, 1.0, [], checked=False)

    data = _safe_parse(raw)
    if data is None:
        return FaithfulnessReport(True, 1.0, [], checked=False)

    return FaithfulnessReport(
        grounded=bool(data.get("grounded", True)),
        faithfulness_score=float(data.get("faithfulness_score", 1.0)),
        unsupported_claims=list(data.get("unsupported_claims", [])),
    )
