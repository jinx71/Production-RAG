import math

from app.bm25_index import BM25Index
from app.embeddings import HashingEmbedder


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def test_hashing_embedder_is_deterministic():
    emb = HashingEmbedder(128)
    assert emb.embed_query("calibration tolerance") == emb.embed_query("calibration tolerance")


def test_hashing_embedder_dimension():
    emb = HashingEmbedder(128)
    assert len(emb.embed_query("anything")) == 128
    assert emb.dimension == 128


def test_lexical_overlap_increases_similarity():
    emb = HashingEmbedder(512)
    base = emb.embed_query("cleaning validation acceptance limit")
    near = emb.embed_query("cleaning validation acceptance criteria limit")
    far = emb.embed_query("audit trail electronic signature login")
    assert _cosine(base, near) > _cosine(base, far)


def test_bm25_ranks_matching_document_first(tmp_path):
    index = BM25Index(str(tmp_path / "bm25.json"))
    index.add(
        ["a", "b", "c"],
        [
            "three consecutive successful cleaning runs are required",
            "calibration tolerance and the four to one accuracy ratio",
            "alcoa attributable legible contemporaneous original accurate",
        ],
    )
    results = index.query("how many cleaning runs are required", top_k=3)
    assert results[0]["id"] == "a"


def test_bm25_persists_and_reloads(tmp_path):
    path = str(tmp_path / "bm25.json")
    index = BM25Index(path)
    index.add(
        ["a", "b", "c"],
        [
            "calibration tolerance accuracy ratio instrument",
            "cleaning validation swab sampling residue",
            "audit trail electronic signature retention",
        ],
    )
    reloaded = BM25Index(path)
    assert reloaded.count() == 3
    assert reloaded.query("calibration tolerance", top_k=1)[0]["id"] == "a"
