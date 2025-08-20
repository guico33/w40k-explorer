from w40k.usecases.answer import AnswerService
from tests.fakes.fake_vector_service import FakeVectorService
from tests.fakes.fake_llm import FakeLLMClient


def test_refusal_content_returns_error_and_no_answer():
    vec_ops = FakeVectorService()
    llm = FakeLLMClient(mode="refusal")

    svc = AnswerService(
        vector_operations=vec_ops,
        llm_client=llm,
        model="fake-model",
        initial_k=2,
        max_context_chunks=2,
        min_similarity_score=0.0,
        max_tokens=128,
    )

    result = svc.answer_query("Disallowed question")

    assert "cannot provide an answer" in result.answer.lower()
    assert result.citations == []
    assert result.confidence == 0.0
