"""Ingestion: load -> chunk -> embed -> index (dense + sparse)."""

import uuid
from pathlib import Path

from app.chunking import chunk_document, load_document
from app.config import Settings

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "scripts" / "seed_documents"


def ingest_text(text: str, source: str, embedder, vectorstore, bm25, settings: Settings) -> int:
    chunks = chunk_document(
        text,
        source=source,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        return 0
    ids = [str(uuid.uuid4()) for _ in chunks]
    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]
    embeddings = embedder.embed_documents(texts)
    vectorstore.add(ids, embeddings, texts, metadatas)
    bm25.add(ids, texts)
    return len(chunks)


def ingest_paths(paths, embedder, vectorstore, bm25, settings: Settings) -> tuple[int, int]:
    documents = 0
    total_chunks = 0
    for path in paths:
        text = load_document(path)
        added = ingest_text(text, Path(path).name, embedder, vectorstore, bm25, settings)
        if added:
            documents += 1
            total_chunks += added
    return documents, total_chunks


def seed_sample_documents(embedder, vectorstore, bm25, settings: Settings) -> tuple[int, int]:
    paths = sorted(str(p) for p in SAMPLE_DIR.glob("*.md"))
    return ingest_paths(paths, embedder, vectorstore, bm25, settings)
