from w40k.usecases.answer import AnswerService
from tests.fakes.fake_vector_ops import FakeVectorOperations
from tests.fakes.fake_llm import FakeLLMClient


def test_threshold_relaxation_recovers_when_no_hits_initially():
    # Fake hits with scores below 0.99 so first pass (min_score=0.99) yields none
    vec_ops = FakeVectorOperations()
    llm = FakeLLMClient(mode="ok")

    svc = AnswerService(
        vector_operations=vec_ops,
        llm_client=llm,
        model="fake-model",
        initial_k=2,
        max_context_chunks=2,
        min_similarity_score=0.99,  # too high for default hits (0.95, 0.90)
        max_tokens=256,
        lower_threshold_on_empty=True,
    )

    result = svc.answer_query("Who is Horus?")

    # Should recover by retrying without min_score and produce an answer
    assert result.answer
    assert vec_ops.queries, "Expected that searches were made"
    # First round has at least one query; second round adds more queries
    assert len(vec_ops.queries) >= 2

