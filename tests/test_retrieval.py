"""Regression tests for hybrid retrieval helpers."""
from langchain_core.documents import Document

from core.retrieval import (
    LexicalIndex,
    SafeRetriever,
    normalize_search_text,
    resolve_follow_up_query,
    search_terms,
)


def test_search_normalization_handles_common_spelling_variants() -> None:
    normalized = normalize_search_text("काठमाण्डौं नेपालको राजधनी हो")
    assert "काठमाडौं" in normalized
    assert "राजधानी" in normalized


def test_search_terms_drop_question_words_and_map_english() -> None:
    assert search_terms("नेपालको राजधानी के हो?") == ["नेपाल", "राजधानी"]
    assert set(search_terms("what is the capital of nepal")) == {"राजधानी", "नेपाल"}


def test_follow_up_query_inherits_previous_user_topic() -> None:
    history = [{"role": "user", "content": "नेपालको राजधानी के हो?"}]
    resolved = resolve_follow_up_query("त्यसको जनसंख्या कति छ?", history)
    assert resolved.startswith("नेपालको राजधानी के हो?")


def test_metadata_survives_name_mutation() -> None:
    document = Document(
        page_content="नेपालको राजधानी काठमाडौं हो।",
        metadata={"source_index": 1, "chunk_index": 0},
    )
    safe = SafeRetriever(retriever_obj=None)
    expanded = safe.expand_documents([document])
    assert expanded[0].metadata["source_index"] == 1


def test_lexical_index_prefers_direct_answer_sentence() -> None:
    documents = [
        Document(
            page_content="राजधानी दैनिकमा प्रकाशित समाचारमा विभिन्न विषय समेटिएका छन्।",
            metadata={"source_index": 1, "chunk_index": 0},
        ),
        Document(
            page_content="काठमाडौं नेपालको राजधानी तथा सबैभन्दा ठूलो शहर हो।",
            metadata={"source_index": 2, "chunk_index": 0},
        ),
    ]
    results = LexicalIndex(documents).search("नेपालको राजधानी के हो?", limit=2)
    assert results[0].metadata["source_index"] == 2


def test_expand_documents_includes_neighbor_context() -> None:
    safe = SafeRetriever(retriever_obj=None)
    safe._source_chunks = {
        7: {
            0: Document(page_content="पहिलो वाक्य।", metadata={"source_index": 7, "chunk_index": 0}),
            1: Document(page_content="दोस्रो वाक्य।", metadata={"source_index": 7, "chunk_index": 1}),
            2: Document(page_content="तेस्रो वाक्य।", metadata={"source_index": 7, "chunk_index": 2}),
            3: Document(page_content="चौथो वाक्य।", metadata={"source_index": 7, "chunk_index": 3}),
        }
    }
    candidate = safe._source_chunks[7][1]
    expanded = safe.expand_documents([candidate], radius=1)
    assert "पहिलो वाक्य" in expanded[0].page_content
    assert "तेस्रो वाक्य" in expanded[0].page_content
    assert "चौथो वाक्य" not in expanded[0].page_content
