from app.chunking import chunk_document


def _long_text() -> str:
    para = (
        "Cleaning validation demonstrates that residues are reduced to acceptable "
        "levels. The most stringent acceptance limit governs the process. "
    )
    return "\n\n".join(para * 3 for _ in range(8))


def test_produces_multiple_chunks():
    chunks = chunk_document(_long_text(), source="doc.md", chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(c.text.strip() for c in chunks)


def test_chunk_size_respected():
    chunks = chunk_document(_long_text(), source="doc.md", chunk_size=300, chunk_overlap=50)
    # allow a chunk to run slightly long only by the overlap carry-over.
    assert all(len(c.text) <= 300 + 50 for c in chunks)


def test_overlap_present_between_consecutive_chunks():
    chunks = chunk_document(_long_text(), source="doc.md", chunk_size=300, chunk_overlap=60)
    texts = [c.text for c in chunks]
    overlaps = 0
    for prev, cur in zip(texts, texts[1:]):
        tail = prev[-30:].strip()
        if tail and tail in cur:
            overlaps += 1
    assert overlaps >= 1


def test_metadata_indices_are_sequential():
    chunks = chunk_document(_long_text(), source="abc.md", chunk_size=300, chunk_overlap=40)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["source"] == "abc.md" for c in chunks)


def test_short_text_single_chunk():
    chunks = chunk_document("Just one short sentence.", source="s.md", chunk_size=300, chunk_overlap=40)
    assert len(chunks) == 1
