from app.guardrails import FaithfulnessReport, check_faithfulness, relevance_gate
from tests.conftest import FakeLLM


def test_relevance_gate_blocks_when_no_results():
    assert relevance_gate(0.9, has_results=False, threshold=0.3) is False


def test_relevance_gate_blocks_below_threshold():
    assert relevance_gate(0.2, has_results=True, threshold=0.3) is False


def test_relevance_gate_passes_above_threshold():
    assert relevance_gate(0.45, has_results=True, threshold=0.3) is True


async def test_faithfulness_parses_grounded_verdict():
    report = await check_faithfulness(FakeLLM(faithful=True), "answer", "context", model="m")
    assert report.grounded is True
    assert report.faithfulness_score == 1.0
    assert report.checked is True


async def test_faithfulness_parses_ungrounded_verdict():
    report = await check_faithfulness(FakeLLM(faithful=False), "answer", "context", model="m")
    assert report.grounded is False
    assert report.faithfulness_score == 0.2


async def test_faithfulness_skips_check_on_refusal():
    refusal = "I don't have enough information in the provided documents to answer that."
    llm = FakeLLM()
    report = await check_faithfulness(llm, refusal, "context", model="m")
    assert report.grounded is True
    assert llm.systems == []  # no LLM call made


async def test_faithfulness_fails_open_on_garbage():
    class GarbageLLM:
        async def complete(self, **kwargs):
            return "not json at all"

    report = await check_faithfulness(GarbageLLM(), "answer", "context", model="m")
    assert isinstance(report, FaithfulnessReport)
    assert report.checked is False
    assert report.grounded is True  # fail open, do not block
