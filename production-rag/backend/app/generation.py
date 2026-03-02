"""Answer generation: context-only, citation-enforced prompting."""

import re

from app.retrieval import RetrievedChunk

REFUSAL = "I don't have enough information in the provided documents to answer that."

_SYSTEM = (
    "You are a precise assistant that answers strictly from the provided context.\n"
    "Rules:\n"
    "- Use ONLY the information in the context blocks. Never use outside knowledge.\n"
    "- Cite every claim with bracketed source numbers like [1], [2] that map to the "
    "context blocks you used.\n"
    f'- If the context does not contain the answer, reply with exactly: "{REFUSAL}"\n'
    "- Be concise and factual; do not speculate."
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        blocks.append(f"[{i}] (source: {source})\n{chunk.text}")
    return "\n\n".join(blocks)


async def generate_answer(
    llm,
    query: str,
    chunks: list[RetrievedChunk],
    *,
    model: str,
) -> tuple[str, list[int]]:
    context = build_context(chunks)
    user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (with citations):"
    answer = (await llm.complete(system=_SYSTEM, user=user, model=model)).strip()
    cited = sorted(
        {int(m) for m in _CITATION_RE.findall(answer) if 1 <= int(m) <= len(chunks)}
    )
    return answer, cited
