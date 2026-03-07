from app.ingest import ingest_text
from tests.conftest import FakeLLM


async def test_pipeline_refuses_on_empty_store(pipeline):
    result = await pipeline.answer("anything unrelated")
    assert result["guardrails"]["refused"] is True
    assert result["guardrails"]["relevance_gate_passed"] is False
    assert "don't have enough information" in result["answer"]


async def test_pipeline_answers_with_citation(settings, embedder, pipeline):
    # seed through the retriever's own stores
    vs = pipeline._retriever._vs
    bm25 = pipeline._retriever._bm25
    ingest_text(
        "A minimum of three consecutive successful cleaning runs is required.",
        "cleaning.md",
        embedder,
        vs,
        bm25,
        settings,
    )
    result = await pipeline.answer("how many consecutive cleaning runs are required")
    assert result["guardrails"]["refused"] is False
    assert result["answer"]
    assert result["citations"]
    assert result["guardrails"]["faithfulness"]["grounded"] is True


async def test_pipeline_flags_unfaithful_answer(settings, embedder, pipeline):
    vs = pipeline._retriever._vs
    bm25 = pipeline._retriever._bm25
    ingest_text("Cleaning validation requires consecutive runs.", "c.md", embedder, vs, bm25, settings)
    pipeline._llm = FakeLLM(faithful=False)
    result = await pipeline.answer("how many cleaning runs are required")
    assert result["guardrails"]["faithfulness"]["grounded"] is False


def test_api_health_and_query(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["data"]["status"] == "ok"

        # empty store -> deterministic refusal, no LLM needed
        refused = client.post("/query", json={"query": "totally unrelated topic"})
        assert refused.json()["data"]["guardrails"]["refused"] is True

        # seed sample docs (hashing embedder, offline) then inject the fake LLM
        seeded = client.post("/ingest/seed")
        assert seeded.json()["data"]["chunks_added"] > 0
        app.state.pipeline._llm = FakeLLM()

        answered = client.post(
            "/query", json={"query": "how many consecutive successful cleaning runs are required"}
        )
        body = answered.json()
        assert body["success"] is True
        assert body["data"]["answer"]
        assert body["data"]["citations"]
