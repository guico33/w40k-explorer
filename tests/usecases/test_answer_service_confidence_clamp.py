from w40k.usecases.answer import AnswerService
from tests.fakes.fake_vector_service import FakeVectorService
from tests.fakes.fake_llm import FakeLLMClient


def test_confidence_is_clamped_between_0_and_1():
    vec_ops = FakeVectorService()
    llm = FakeLLMClient(mode="bad_conf")

    svc = AnswerService(
        vector_operations=vec_ops,
        llm_client=llm,
        model="fake-model",
        initial_k=2,
        max_context_chunks=2,
        min_similarity_score=0.0,
        max_tokens=128,
    )

    result = svc.answer_query("Who is Horus?")

    assert 0.0 <= result.confidence <= 1.0
