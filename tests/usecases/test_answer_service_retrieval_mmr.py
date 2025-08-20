from w40k.usecases.answer import AnswerService
from tests.fakes.fake_llm import FakeLLMClient
from tests.fakes.fake_vector_service import FakeVectorService


def test_diversify_mmr_limits_duplicates_and_article_repeats():
    # Craft overlapping hits: same article with multiple sections; ensure de-dup
    hits = [
        {
            "chunk_uid": f"a-{i}",
            "article_title": "Horus",
            "canonical_url": "https://example.org/horus",
            "section_path": ["Biography", "Early" if i % 2 == 0 else "Later"],
            "lead": i == 0,
            "kv_data": {},
            "text": f"chunk {i}",
            "score": 1.0 - i * 0.01,
        }
        for i in range(6)
    ]
    fake_vec = FakeVectorService(hits=hits)
    svc = AnswerService(
        vector_operations=fake_vec,
        llm_client=FakeLLMClient(mode="ok"),
        model="fake-model",
        initial_k=6,
        max_context_chunks=4,
        min_similarity_score=0.0,
        max_tokens=128,
    )

    diversified = svc._diversify_mmr(hits, max_chunks=4)

    # Should not exceed max chunks
    assert len(diversified) <= 4
    # Should not repeat the same (canonical_url, first two section levels)
    keys = set()
    for h in diversified:
        section = h.get("section_path", [])
        if isinstance(section, list):
            section_key = tuple(section[:2])
        else:
            section_key = ()
        key = (h.get("canonical_url"), section_key)
        assert key not in keys
        keys.add(key)
