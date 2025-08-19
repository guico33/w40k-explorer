from w40k.usecases.answer import AnswerService
from tests.fakes.fake_vector_ops import FakeVectorOperations
from tests.fakes.fake_llm import FakeLLMClient


def test_answer_service_non_json_fallback_has_no_citations_and_sets_error():
    vec_ops = FakeVectorOperations()
    llm = FakeLLMClient(mode="non_json")

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

    assert result.answer, "Should still return an answer string"
    assert result.citations == [], "Non-JSON fallback should not build citations"
    assert result.error is not None, "Expected an error flag for non-JSON response"

