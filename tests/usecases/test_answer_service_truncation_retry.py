from w40k.usecases.answer import AnswerService
from tests.fakes.fake_vector_ops import FakeVectorOperations
from tests.fakes.fake_llm import FakeLLMClient


def test_answer_service_truncation_triggers_compression_retry_and_succeeds():
    vec_ops = FakeVectorOperations()
    llm = FakeLLMClient(mode="incomplete")

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

    assert result.answer.startswith("Compressed:"), "Expected compressed retry answer"
    assert result.citations, "Citations should be present after retry"

