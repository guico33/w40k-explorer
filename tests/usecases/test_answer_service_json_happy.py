from w40k.usecases.answer import AnswerService
from tests.fakes.fake_vector_service import FakeVectorService
from tests.fakes.fake_llm import FakeLLMClient


def test_answer_service_parses_json_and_builds_citations():
    vec_ops = FakeVectorService()
    llm = FakeLLMClient(mode="ok")

    svc = AnswerService(
        vector_operations=vec_ops,
        llm_client=llm,
        model="fake-model",
        initial_k=2,
        max_context_chunks=2,
        min_similarity_score=0.0,
        max_tokens=256,
    )

    result = svc.answer_query("Who is Horus?")

    assert result.answer, "Expected non-empty answer"
    assert result.citations, "Expected citations built from citations_used"
    # Verify first citation maps to context id 0 and contains required fields
    c0 = result.citations[0]
    assert c0["id"] == 0
    assert "title" in c0 and "section" in c0 and "url" in c0
    # Confidence should be within 0-1 bounds
    assert 0.0 <= result.confidence <= 1.0
