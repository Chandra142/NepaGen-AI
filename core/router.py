"""Query classification and streaming response generation for NepaGen AI."""
from __future__ import annotations
import re
from typing import Callable, Generator

import streamlit as st

from core.config import (
    CONVERSATIONAL_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    NONSENSE_RESPONSE,
    GROQ_ERROR_RESPONSE,
    GREETING_MAP,
    CASUAL_SHORT_REPLIES,
    QUESTION_PREFIXES,
    PERSONAL_PATTERNS,
)
from core.memory import normalized_compact_text, recent_chat_context
from core.retrieval import (
    expand_documents,
    normalize_search_text,
    resolve_follow_up_query,
    safe_retrieve,
    search_terms,
)


# ─── Noise / Nonsense Detectors ───────────────────────────────────────────────

def _is_nonsense_query(compact: str) -> bool:
    if not compact:
        return True
    if re.fullmatch(r"[0-9\s]+", compact):
        return True
    if re.fullmatch(r"[a-z]+", compact) and len(compact) >= 4 and not re.search(r"[aeiou]", compact):
        return True
    if re.search(r"(.)\1{6,}", compact):
        return True
    if len(compact) <= 2 and compact not in {"ok", "no", "yes", "hmm"}:
        return True
    return False


def _is_noise_doc(text: str) -> bool:
    """Filter out low-quality retrieved chunks conservatively."""
    low = text.lower()
    if "read more" in low or "imagekhabar" in low or "--instant articles--" in low:
        return True
    if re.search(r"(.)\1{10,}", text):
        return True
    if len(text.strip()) < 30:
        return True
    if re.fullmatch(r"https?://[^\s]+", text.strip()):
        return True
    return False


def _sanitize_html(text: str) -> str:
    """Remove HTML, CSS, and JS from retrieved chunks."""
    text = re.sub(r'(?is)<(script|style|iframe|svg)[^>]*>.*?</\1>', ' ', text)
    text = re.sub(r'(?s)<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─── Query Classifier ─────────────────────────────────────────────────────────

CASUAL_SMALL_TALK: frozenset[str] = frozenset(
    {
        "how are you", "how r you", "how r u", "what's up", "whats up",
        "k xa", "के छ", "कस्तो छ",
        "who are you", "what are you", "what is your name",
        "what can you do", "tell me about yourself",
        "what is my name", "what's my name",
        "तपाईंको नाम के हो", "तिम्रो नाम के हो", "मेरो नाम के हो",
        "तपाईं को हुनुहुन्छ", "तिमी को हौ", "तपाईंलाई कस्तो छ", "तिमीलाई कस्तो छ",
    }
)

SMALL_TALK_PREFIXES: tuple[str, ...] = (
    "how are you", "how r you", "how r u", "what's up", "whats up",
    "k xa", "के छ", "कस्तो छ", "तपाईंलाई कस्तो छ", "तिमीलाई कस्तो छ",
)

GREETING_PREFIXES: tuple[str, ...] = (
    "hi", "hello", "hey", "namaste", "नमस्ते", "yo",
    "good morning", "good evening", "good night",
    "bye", "thanks", "thank you",
)

ENGLISH_QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "what", "who", "where", "when", "why", "how", "which",
        "define", "explain", "describe", "meaning", "history", "list",
    }
)

# Devanagari question words attach postpositions ("कतिको", "कहाँबाट"), so we
# match these by prefix. "के" is matched as a whole word to avoid false hits
# inside words such as "केही" and "केटा".
NEPALI_QUESTION_PREFIXES: tuple[str, ...] = (
    "कति", "कहाँ", "कहिले", "कुन", "कसरी", "किन", "कसले", "कस्तो",
)

NEPALI_INFORMATION_MARKERS: tuple[str, ...] = (
    "बारेमा", "बताउनुहोस्", "भन्नुहोस्", "जानकारी", "परिचय",
)


def _starts_with_phrase(compact: str, phrases: tuple[str, ...]) -> bool:
    return any(compact == phrase or compact.startswith(phrase + " ") for phrase in phrases)


def _is_question_like(raw_query: str, compact: str, words: list[str]) -> bool:
    """Detect informational intent from the original, unnormalized query.

    ``compact`` has punctuation stripped, so the question mark has to be read
    from ``raw_query``. The old check on ``compact.endswith("?")`` could never
    be true and made question-marked queries look like statements.
    """
    if "?" in raw_query:
        return True
    if any(word in ENGLISH_QUESTION_WORDS for word in words):
        return True
    if "के" in words:
        return True
    if any(word.startswith(prefix) for word in words for prefix in NEPALI_QUESTION_PREFIXES):
        return True
    if any(marker in compact for marker in NEPALI_INFORMATION_MARKERS):
        return True
    return any(compact.startswith(prefix) for prefix in QUESTION_PREFIXES)


def classify_query(query: str) -> str:
    """Return a routing category for *query*.

    Casual intents are checked before the nonsense filter so short greetings
    such as "hi" are not rejected as noise. Anything that is neither clearly
    casual nor personal defaults to retrieval-backed answering, so the
    assistant does not improvise facts from the conversational model.
    """
    raw_query = query.strip().lower()
    compact = normalized_compact_text(query)
    words = compact.split()
    question_like = _is_question_like(raw_query, compact, words)

    if compact in GREETING_MAP or compact in CASUAL_SHORT_REPLIES:
        return "casual_chat"
    if compact in CASUAL_SMALL_TALK or _starts_with_phrase(compact, SMALL_TALK_PREFIXES):
        return "casual_chat"
    if (
        _starts_with_phrase(compact, GREETING_PREFIXES)
        and any(phrase in compact for phrase in SMALL_TALK_PREFIXES)
    ):
        return "casual_chat"
    if (
        len(words) <= 5
        and not question_like
        and _starts_with_phrase(compact, GREETING_PREFIXES)
    ):
        return "casual_chat"

    if _is_nonsense_query(compact):
        return "nonsense"

    if any(re.match(pattern, compact) for pattern in PERSONAL_PATTERNS):
        return "personal_statement"

    if question_like:
        grounded_terms = ("नेपाल", "nepal", "rag")
        return "RAG_query" if any(term in compact for term in grounded_terms) else "factual_question"

    # Longer free-form text is treated as an information request; short
    # unknown utterances stay conversational.
    if len(words) > 2:
        return "RAG_query"
    return "casual_chat"


# ─── Prompt Builders ──────────────────────────────────────────────────────────

def _conversational_prompt(query: str) -> str:
    profile = st.session_state.get("user_profile", {})
    known_name = profile.get("name")
    ctx = recent_chat_context(exclude_latest=True)
    return (
        f"{CONVERSATIONAL_SYSTEM_PROMPT}\n\n"
        f"Known user info: {f'Name: {known_name}' if known_name else 'None'}\n\n"
        f"Recent conversation:\n{ctx if ctx else 'No recent conversation context.'}\n\n"
        f"User message:\n{query}\n\n"
        f"Assistant reply:"
    )


def _rag_prompt(query: str, context: str) -> str:
    ctx = recent_chat_context(exclude_latest=True)
    return (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"--- UNTRUSTED RETRIEVED CONTEXT START ---\n"
        f"{context}\n"
        f"--- UNTRUSTED RETRIEVED CONTEXT END ---\n\n"
        f"Recent conversation:\n{ctx if ctx else 'No recent conversation context.'}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"Correct Answer in Nepali (no HTML/CSS/code):"
    )


def _prepare_context(query: str, retriever) -> tuple[str, list]:
    """Retrieve, expand, and rank complete source windows for the prompt."""
    search_query = resolve_follow_up_query(query, st.session_state.get("messages", []))
    candidates = safe_retrieve(retriever, search_query)
    documents = expand_documents(retriever, candidates)

    for document in documents:
        document.page_content = _sanitize_html(document.page_content)
    documents = [doc for doc in documents if not _is_noise_doc(doc.page_content)]

    terms = search_terms(search_query)
    scored_documents = []
    for document in documents:
        normalized_document = normalize_search_text(document.page_content)
        overlap = sum(1 for term in terms if term in normalized_document)
        hybrid_score = float(document.metadata.get("hybrid_score", 0))
        lexical_score = float(document.metadata.get("lexical_score", 0))
        quality_score = float(document.metadata.get("quality_score", 0))
        score = overlap + lexical_score * 0.3 + hybrid_score * 5.0 + quality_score * 0.15
        scored_documents.append((score, document))

    scored_documents.sort(key=lambda item: item[0], reverse=True)
    top_documents = [document for score, document in scored_documents[:5]]
    context = "\n\n".join(
        f"[Reference {index}]\n{document.page_content}"
        for index, document in enumerate(top_documents, start=1)
    )
    return context, top_documents


# ─── Streaming Response Generator ─────────────────────────────────────────────

def stream_reply(
    query: str,
    category: str,
    retriever,
    llm,
    on_sources: Callable[[list], None] | None = None,
) -> Generator[str, None, None]:
    """Yield response text chunks and optionally publish retrieved sources."""

    if category == "nonsense":
        yield NONSENSE_RESPONSE
        return

    compact = normalized_compact_text(query)

    # ── Instant lookups (no LLM) ──────────────────────────────────────────────
    if category in {"greeting", "casual_chat"}:
        if compact in GREETING_MAP:
            yield GREETING_MAP[compact]
            return
        if compact in CASUAL_SHORT_REPLIES:
            yield CASUAL_SHORT_REPLIES[compact]
            return
        how_are_you = {"how are you", "how r you", "how r u", "what's up", "whats up", "k xa", "के छ", "कस्तो छ"}
        if compact in how_are_you:
            yield "म राम्रो छु 😊 तपाईंलाई कसरी सहयोग गरौं?"
            return

    import traceback
    # ── Conversational / personal ─────────────────────────────────────────────
    if category in {"greeting", "casual_chat", "personal_statement"}:
        prompt = _conversational_prompt(query)
        try:
            for chunk in llm.stream(prompt):
                yield chunk.content or ""
        except Exception as exc:
            print(f"[router] LLM stream error (conversational): {exc!r}")
            traceback.print_exc()
            yield GROQ_ERROR_RESPONSE
        return

    # ── RAG / factual ─────────────────────────────────────────────────────────
    if category in {"factual_question", "RAG_query"}:
        context, sources = _prepare_context(query, retriever)
        if on_sources:
            on_sources(sources)
        if not context.strip():
            yield "मलाई जानकारी भेटिएन। कृपया प्रश्नलाई अलि फरक तरिकाले सोध्नुहोस्।"
            return
        prompt = _rag_prompt(query, context)
        try:
            for chunk in llm.stream(prompt):
                yield chunk.content or ""
        except Exception as exc:
            print(f"[router] LLM stream error (RAG): {exc!r}")
            traceback.print_exc()
            yield GROQ_ERROR_RESPONSE
        return

    # ── Default fallback ──────────────────────────────────────────────────────
    prompt = _conversational_prompt(query)
    try:
        for chunk in llm.stream(prompt):
            yield chunk.content or ""
    except Exception as exc:
        print(f"[router] LLM stream error (fallback): {exc!r}")
        traceback.print_exc()
        yield GROQ_ERROR_RESPONSE
