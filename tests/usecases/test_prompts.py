from w40k.usecases.prompts import (
    build_system_prompt,
    build_compressed_prompt,
    build_query_expansion_prompt,
)


def test_build_system_prompt_interpolates_parameters():
    max_sentences = 5
    max_chars = 1500
    
    prompt = build_system_prompt(max_sentences=max_sentences, max_chars=max_chars)
    
    assert str(max_sentences) in prompt
    assert str(max_chars) in prompt


def test_build_system_prompt_returns_non_empty_string():
    prompt = build_system_prompt()
    
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_compressed_prompt_interpolates_max_sentences():
    max_sentences = 2
    
    prompt = build_compressed_prompt(max_sentences=max_sentences)
    
    assert str(max_sentences) in prompt


def test_build_query_expansion_prompt_includes_question_and_count():
    question = "Who is the Emperor?"
    n = 3
    
    prompt = build_query_expansion_prompt(question=question, n=n)
    
    assert question in prompt
    assert str(n) in prompt