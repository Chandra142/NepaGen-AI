"""Regression tests for intent routing."""
import pytest

from core.router import classify_query


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # Greetings and casual chat (previously "hi" was rejected as nonsense)
        ("hi", "casual_chat"),
        ("hello", "casual_chat"),
        ("नमस्ते", "casual_chat"),
        ("thanks", "casual_chat"),
        ("how are you", "casual_chat"),
        ("hello, how are you?", "casual_chat"),
        ("good morning sir", "casual_chat"),
        # The app's own suggestion chips must reach retrieval
        ("सगरमाथाको बारेमा बताउनुहोस्।", "factual_question"),
        ("सगरमाथाको उचाइ कति छ?", "factual_question"),
        ("नेपालको संस्कृति के हो?", "RAG_query"),
        ("नेपालको इतिहासबारे बताउनुहोस्।", "RAG_query"),
        ("नेपालका प्रमुख पर्वहरू के के हुन्?", "RAG_query"),
        ("नेपालको राजधानी", "RAG_query"),
        ("what is the capital of nepal", "RAG_query"),
        ("tell me about mount everest", "factual_question"),
        # Personal statements stay conversational
        ("i am a student", "personal_statement"),
        ("my name is ram", "personal_statement"),
        # Noise handling
        ("12345", "nonsense"),
        ("", "nonsense"),
        ("apple", "casual_chat"),
    ],
)
def test_classify_query(query: str, expected: str) -> None:
    assert classify_query(query) == expected


def test_question_mark_is_preserved_for_detection() -> None:
    """Normalization strips '?', so detection must read the raw query."""
    assert classify_query("सगरमाथाको उचाइ कति छ?") in {"RAG_query", "factual_question"}
