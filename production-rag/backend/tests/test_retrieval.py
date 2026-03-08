from app.retrieval import HybridRetriever, reciprocal_rank_fusion


def test_rrf_rewards_agreement_across_both_lists():
    dense = [
        {"id": "x", "text": "x", "metadata": {}, "score": 0.9},
        {"id": "y", "text": "y", "metadata": {}, "score": 0.8},
    ]
    sparse = [
        {"id": "x", "text": "x", "metadata": {}, "score": 5.0},
        {"id": "z", "text": "z", "metadata": {}, "score": 4.0},
    ]
    fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=3)
    # x appears top of both lists -> must rank first.
    assert fused[0].id == "x"
    assert fused[0].dense_rank == 0 and fused[0].sparse_rank == 0


def test_rrf_includes_items_from_either_list():
    dense = [{"id": "a", "text": "a", "metadata": {}, "score": 0.5}]
    sparse = [{"id": "b", "text": "b", "metadata": {}, "score": 2.0}]
    fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=5)
    ids = {c.id for c in fused}
    assert ids == {"a", "b"}
    only_a = next(c for c in fused if c.id == "a")
    assert only_a.dense_rank == 0 and only_a.sparse_rank is None


def test_hybrid_retriever_finds_relevant_chunk(settings, embedder, seed_corpus):
    vectorstore, bm25 = seed_corpus
    retriever = HybridRetriever(
        vectorstore, bm25, embedder, candidate_k=settings.candidate_k, rrf_k=settings.rrf_k
    )
    chunks, top_dense = retriever.retrieve("how many consecutive cleaning runs are required", top_k=3)
    assert chunks
    assert any("consecutive" in c.text for c in chunks)
    assert top_dense > 0.0


def test_retriever_returns_empty_on_empty_store(settings, embedder, stores):
    vectorstore, bm25 = stores
    retriever = HybridRetriever(
        vectorstore, bm25, embedder, candidate_k=settings.candidate_k, rrf_k=settings.rrf_k
    )
    chunks, top_dense = retriever.retrieve("anything at all", top_k=3)
    assert chunks == []
    assert top_dense == 0.0
