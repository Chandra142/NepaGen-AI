"""Hybrid retrieval and context-window helpers for NepaGen AI.

Vector search is combined with a lightweight multilingual lexical index. Nepali
news text often uses spelling variants and the answer can be in a neighboring
chunk, so retrieval normalizes common variants and expands source windows.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import unicodedata

from langchain_core.documents import Document


# ─── Search text normalization ────────────────────────────────────────────────

_ASCII_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("prime minister", "प्रधानमन्त्री"),
    ("national animal", "राष्ट्रिय जनावर"),
    ("mount everest", "सगरमाथा"),
    ("kathmandu", "काठमाडौं"),
    ("sagarmatha", "सगरमाथा"),
    ("everest", "सगरमाथा"),
    ("rajdhani", "राजधानी"),
    ("capital", "राजधानी"),
    ("history", "इतिहास"),
    ("culture", "संस्कृति"),
    ("population", "जनसंख्या"),
    ("provinces", "प्रदेश"),
    ("province", "प्रदेश"),
    ("states", "प्रदेश"),
    ("state", "प्रदेश"),
    ("festivals", "पर्व"),
    ("festival", "पर्व"),
    ("height", "उचाइ"),
    ("nepal", "नेपाल"),
)

_NEPALI_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("काठमाण्डौ", "काठमाडौं"),
    ("काठमांडौ", "काठमाडौं"),
    ("काठमान्डु", "काठमाडौं"),
    ("काठमाडौँ", "काठमाडौं"),
    ("राजधनी", "राजधानी"),
    ("चाडपर्व", "पर्व"),
    ("चाड", "पर्व"),
    ("जन्तु", "जनावर"),
    ("पशु", "जनावर"),
    ("उचाई", "उचाइ"),
    ("अग्लो", "उचाइ"),
    ("अग्ला", "उचाइ"),
)

_ASCII_MAP = dict(_ASCII_REPLACEMENTS)
_NEPALI_MAP = dict(_NEPALI_REPLACEMENTS)
_ASCII_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(source) for source, target in _ASCII_REPLACEMENTS) + r")\b"
)
_NEPALI_PATTERN = re.compile(
    "|".join(re.escape(source) for source, target in _NEPALI_REPLACEMENTS)
)

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "who", "where",
    "when", "why", "how", "many", "much", "tell", "me", "about", "of", "in",
    "on", "at", "to", "for", "and", "or", "please", "can", "you", "your", "do",
    "does", "did", "it", "its", "this", "that", "these", "those", "then",
    "के", "हो", "हुन्", "छ", "छन्", "कति", "कहाँ", "कहिले", "कुन", "किन",
    "कसरी", "कसले", "बारेमा", "बताउनुहोस्", "भन्नुहोस्", "जानकारी", "दिनुहोस्",
    "मलाई", "को", "का", "की", "लाई", "मा", "म", "यो", "त्यो", "ती", "र", "वा",
    "पनि", "समेत", "गर्नुहोस्", "बारे", "कृपया", "तपाईं", "तिमी", "मेरो", "तिम्रो",
    "प्रमुख", "महान",
}

_SUFFIXES = (
    "हरूलाई", "हरूको", "हरूबाट", "हरूमा", "हरू", "लाई", "बाट", "सँग", "संग",
    "को", "का", "की", "के", "ले", "मा",
)

_NUMBER_MARKERS = (
    "कति", "उचाइ", "जनसंख्या", "लम्बाइ", "चौडाइ", "क्षेत्रफल", "उमेर",
    "how many", "height", "population", "old",
)

_FOLLOW_UP_MARKERS = (
    "त्यसको", "यसको", "उसको", "त्यस", "यस", "उनको", "त्यहाँ", "यहाँ",
    "its", "their", "there",
)


def normalize_search_text(text: str) -> str:
    """Normalize spelling variants for matching, not for display."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _ASCII_PATTERN.sub(lambda match: _ASCII_MAP[match.group(0)], text)
    text = _NEPALI_PATTERN.sub(lambda match: _NEPALI_MAP[match.group(0)], text)
    text = re.sub(r"[^\wऀ-ॿ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def search_terms(query: str) -> list[str]:
    """Return distinctive, normalized content terms from a question."""
    terms = []
    for token in normalize_search_text(query).split():
        term = _stem(token)
        if term == "राज्य":
            term = "प्रदेश"
        if len(term) > 1 and term not in _STOPWORDS and term not in terms:
            terms.append(term)
    return terms


def semantic_query(query: str) -> str:
    """Remove conversational/question wording before a second vector search."""
    terms = search_terms(query)
    if len(terms) == 1 and terms[0] != "नेपाल":
        return f"{terms[0]} नेपाल"
    return " ".join(terms) or normalize_search_text(query)


def resolve_follow_up_query(query: str, history: list[dict]) -> str:
    """Add the previous user topic to a pronoun-based follow-up question."""
    compact = normalize_search_text(query)
    if not any(marker in compact for marker in _FOLLOW_UP_MARKERS):
        return query

    previous = history
    if history and history[-1].get("role") == "user":
        last_user = normalize_search_text(history[-1].get("content", ""))
        if last_user == compact:
            previous = history[:-1]
    previous_users = [item["content"] for item in previous if item.get("role") == "user"]
    if not previous_users:
        return query
    return f"{previous_users[-1]} {query}"


def _terms_near(text: str, terms: list[str], window: int = 120) -> bool:
    """Return whether all query terms occur within one short passage."""
    if len(terms) <= 1:
        return True
    positions = [
        [match.start() for match in re.finditer(re.escape(term), text)]
        for term in terms
    ]
    if any(not term_positions for term_positions in positions):
        return False
    return any(
        all(any(abs(start - other) <= window for other in term_positions) for term_positions in positions[1:])
        for start in positions[0]
    )


def _answer_sentence_bonus(text: str, terms: list[str], expects_number: bool) -> float:
    """Prefer short passages that contain a complete, answer-shaped statement."""
    best = 0.0
    for sentence in re.split(r"[।!?]+|\n+", text):
        if not all(term in sentence for term in terms):
            continue
        score = 1.5
        if expects_number and re.search(r"[0-9०-९]", sentence):
            score += 4.0
            if any(marker in sentence for marker in ("विश्व", "संसार")):
                score += 2.5
        if len(terms) == 1 and any(
            marker in sentence
            for marker in ("हो", "पर्छ", "रहेको", "जिल्ला", "शहर", "राजधानी", "इतिहास", "परिचय")
        ):
            score += 2.0
        best = max(best, score)
    return best


def _expects_number(query: str) -> bool:
    normalized = normalize_search_text(query)
    return any(marker in normalized for marker in _NUMBER_MARKERS)


def _number_near(text: str, terms: list[str], window: int = 64) -> bool:
    numbers = list(re.finditer(r"[0-9०-९]+(?:[.,][0-9०-९]+)?", text))
    if not numbers:
        return False
    for term in terms:
        for match in re.finditer(re.escape(term), text):
            if any(
                number.end() >= match.start() - window
                and number.start() <= match.end() + window
                for number in numbers
            ):
                return True
    return False


class LexicalIndex:
    """Small in-memory keyword index over the existing FAISS docstore."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.texts = [normalize_search_text(doc.page_content) for doc in documents]

    def search(self, query: str, limit: int = 20) -> list[Document]:
        terms = search_terms(query)
        if not terms:
            return []

        matches = []
        document_frequency: Counter[str] = Counter()
        for index, text in enumerate(self.texts):
            counts = {term: text.count(term) for term in terms}
            counts = {term: count for term, count in counts.items() if count}
            if not counts:
                continue
            document_frequency.update(counts.keys())
            matches.append((index, text, counts))

        total_documents = max(1, len(self.texts))
        idf = {
            term: max(0.5, math.log((total_documents + 1) / (document_frequency[term] + 1)))
            for term in terms
        }
        total_weight = sum(idf.values()) or 1.0
        expects_number = _expects_number(query)
        scored = []

        for index, text, counts in matches:
            coverage = sum(idf[term] for term in counts) / total_weight
            frequency_score = sum(idf[term] * min(count, 3) * 0.25 for term, count in counts.items())
            score = coverage * 6.0 + frequency_score
            if _terms_near(text, list(counts)):
                score += 3.0 * coverage
            elif len(counts) < len(terms):
                score *= 0.6
            score += _answer_sentence_bonus(text, list(counts), expects_number) * coverage
            if expects_number:
                if _number_near(text, list(counts)):
                    score += 2.5 * coverage
                elif not re.search(r"[0-9०-९]", text):
                    score *= 0.5
            quality = float(self.documents[index].metadata.get("quality_score", 0))
            score += quality * 0.15
            scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        results = []
        for score, index in scored[:limit]:
            document = self.documents[index]
            document.metadata["lexical_score"] = round(score, 4)
            results.append(document)
        return results


# ─── Retriever wrappers ───────────────────────────────────────────────────────

class _EmptyRetriever:
    """No-op retriever used when the vectorstore is unavailable."""

    def get_relevant_documents(self, query: str) -> list:  # noqa: ARG002
        return []

    def invoke(self, query: str) -> list:  # noqa: ARG002
        return []

    def hybrid_retrieve(self, query: str, limit: int = 24) -> list:  # noqa: ARG002
        return []

    def expand_documents(self, documents: list[Document], radius: int = 2, max_sources: int = 10) -> list[Document]:  # noqa: ARG002
        return documents


class SafeRetriever:
    """Wrap a LangChain retriever with safe vector and lexical retrieval."""

    def __init__(self, retriever_obj, vectorstore=None, similarity_retriever=None) -> None:
        self._retriever = retriever_obj
        self.vectorstore = vectorstore
        self._similarity_retriever = similarity_retriever
        self._lexical_index: LexicalIndex | None = None
        self._source_chunks: dict[int, dict[int, Document]] | None = None

    def get_relevant_documents(self, query: str) -> list:
        try:
            return self._retriever.get_relevant_documents(query)
        except Exception as exc:
            print(f"[retrieval] get_relevant_documents failed: {exc!r}")
            return []

    def invoke(self, query: str) -> list:
        try:
            return self._retriever.invoke(query)
        except Exception as exc:
            print(f"[retrieval] invoke failed: {exc!r}")
            return []

    def _all_documents(self) -> list[Document]:
        if self.vectorstore is None:
            return []
        documents = []
        for doc_id in self.vectorstore.index_to_docstore_id.values():
            document = self.vectorstore.docstore.search(doc_id)
            if isinstance(document, Document):
                documents.append(document)
        return documents

    def _get_lexical_index(self) -> LexicalIndex:
        if self._lexical_index is None:
            self._lexical_index = LexicalIndex(self._all_documents())
        return self._lexical_index

    @staticmethod
    def _document_key(document: Document) -> tuple:
        metadata = document.metadata
        if "source_index" in metadata and "chunk_index" in metadata:
            return metadata["source_index"], metadata["chunk_index"]
        return hash(document.page_content)

    def hybrid_retrieve(self, query: str, limit: int = 24) -> list[Document]:
        """Fuse MMR, normalized similarity, and lexical candidate rankings."""
        rankings: list[tuple[float, list[Document]]] = [(1.0, self.invoke(query))]

        if self._similarity_retriever is not None:
            try:
                rankings.append((1.15, self._similarity_retriever.invoke(semantic_query(query))))
            except Exception as exc:
                print(f"[retrieval] normalized similarity failed: {exc!r}")

        if self.vectorstore is not None:
            try:
                rankings.append((1.7, self._get_lexical_index().search(query, limit=20)))
            except Exception as exc:
                print(f"[retrieval] lexical search failed: {exc!r}")

        scores: dict[tuple, float] = defaultdict(float)
        documents: dict[tuple, Document] = {}
        for weight, ranking in rankings:
            for rank, document in enumerate(ranking, start=1):
                key = self._document_key(document)
                scores[key] += weight / (rank + 8.0)
                documents[key] = document

        ordered = sorted(scores, key=lambda key: scores[key], reverse=True)[:limit]
        results = []
        for key in ordered:
            document = documents[key]
            document.metadata["hybrid_score"] = round(scores[key], 5)
            results.append(document)
        return results

    def _get_source_chunks(self) -> dict[int, dict[int, Document]]:
        if self._source_chunks is None:
            source_chunks: dict[int, dict[int, Document]] = defaultdict(dict)
            for document in self._all_documents():
                source = document.metadata.get("source_index")
                chunk = document.metadata.get("chunk_index")
                if source is not None and chunk is not None:
                    source_chunks[source][chunk] = document
            self._source_chunks = source_chunks
        return self._source_chunks

    @staticmethod
    def _merge_chunks(documents: list[Document]) -> str:
        """Join adjacent chunks without repeating their overlap text."""
        merged = ""
        for document in documents:
            part = document.page_content.strip()
            if not part:
                continue
            if not merged:
                merged = part
                continue
            overlap = 0
            for size in range(min(96, len(merged), len(part)), 15, -1):
                if merged.endswith(part[:size]):
                    overlap = size
                    break
            merged += part[overlap:] if overlap else " " + part
        return merged

    def expand_documents(self, documents: list[Document], radius: int = 2, max_sources: int = 10) -> list[Document]:
        """Add neighboring chunks and keep at most one window per source."""
        source_chunks = self._get_source_chunks()
        grouped: dict[int, list[Document]] = defaultdict(list)
        unscoped = []
        for document in documents:
            source = document.metadata.get("source_index")
            chunk = document.metadata.get("chunk_index")
            if source is None or chunk is None or source not in source_chunks:
                unscoped.append(document)
            else:
                grouped[source].append(document)

        ordered_sources = sorted(
            grouped,
            key=lambda source: max(
                float(document.metadata.get("hybrid_score", 0))
                for document in grouped[source]
            ),
            reverse=True,
        )
        expanded = list(unscoped)
        for source in ordered_sources:
            best_document = max(
                grouped[source],
                key=lambda document: (
                    float(document.metadata.get("lexical_score", 0)),
                    float(document.metadata.get("hybrid_score", 0)),
                ),
            )
            chunk = best_document.metadata["chunk_index"]
            neighbors = [
                source_chunks[source][index]
                for index in range(chunk - radius, chunk + radius + 1)
                if index in source_chunks[source]
            ]
            metadata = dict(best_document.metadata)
            metadata["window_start"] = max(0, chunk - radius)
            metadata["window_end"] = chunk + radius
            expanded.append(
                Document(page_content=self._merge_chunks(neighbors), metadata=metadata)
            )
            if len(expanded) >= max_sources:
                break
        return expanded[:max_sources]


def make_retriever(db, *, k: int = 16, fetch_k: int = 80) -> SafeRetriever | _EmptyRetriever:
    """Return a hybrid retriever, or a no-op retriever for an empty index."""
    if getattr(db.index, "ntotal", 0) > 0:
        return SafeRetriever(
            db.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k}),
            vectorstore=db,
            similarity_retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 24}),
        )
    print("[retrieval] empty vectorstore — using no-op retriever")
    return _EmptyRetriever()


def safe_retrieve(retriever, query: str) -> list[Document]:
    """Return fused retrieval candidates with a compatibility fallback."""
    if hasattr(retriever, "hybrid_retrieve"):
        return retriever.hybrid_retrieve(query)
    return retriever.invoke(query)


def expand_documents(retriever, documents: list[Document]) -> list[Document]:
    """Expand candidate chunks into readable source windows when possible."""
    if hasattr(retriever, "expand_documents"):
        return retriever.expand_documents(documents)
    return documents
