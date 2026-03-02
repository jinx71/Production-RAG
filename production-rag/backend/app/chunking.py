"""Document loading and recursive character chunking.

The splitter walks a separator hierarchy (paragraph -> line -> sentence -> word
-> char) so each atom stays within `chunk_size`, then greedily packs atoms into
chunks and carries a trailing window forward to create overlap.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def load_document(path: str | Path) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        import fitz  # PyMuPDF, local import so the module loads without it

        doc = fitz.open(p)
        try:
            return "\n\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
    return p.read_text(encoding="utf-8")


def _atomic_split(text: str, chunk_size: int, separators: list[str] = _SEPARATORS) -> list[str]:
    """Recursively split until every piece is <= chunk_size (best effort).

    Separators are re-attached to keep the concatenation faithful to the source.
    """
    separator = separators[-1]
    rest: list[str] = []
    for i, sep in enumerate(separators):
        if sep == "":
            separator, rest = "", []
            break
        if sep in text:
            separator, rest = sep, separators[i + 1:]
            break

    if separator == "":
        pieces = list(text)
    else:
        parts = text.split(separator)
        pieces = [part + separator for part in parts[:-1]]
        if parts[-1] != "":
            pieces.append(parts[-1])

    atoms: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size or not rest:
            atoms.append(piece)
        else:
            atoms.extend(_atomic_split(piece, chunk_size, rest))
    return [a for a in atoms if a != ""]


def _merge_with_overlap(atoms: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for atom in atoms:
        if current and current_len + len(atom) > chunk_size:
            chunks.append("".join(current))
            if overlap > 0:
                carry: list[str] = []
                carry_len = 0
                for prev in reversed(current):
                    if carry_len + len(prev) > overlap:
                        break
                    carry.insert(0, prev)
                    carry_len += len(prev)
                current, current_len = carry, carry_len
            else:
                current, current_len = [], 0
        current.append(atom)
        current_len += len(atom)

    if current:
        chunks.append("".join(current))
    return [c for c in chunks if c.strip()]


def chunk_document(
    text: str,
    *,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    atoms = _atomic_split(text, chunk_size)
    pieces = _merge_with_overlap(atoms, chunk_size, chunk_overlap)
    return [
        Chunk(text=piece.strip(), metadata={"source": source, "chunk_index": i})
        for i, piece in enumerate(pieces)
    ]
