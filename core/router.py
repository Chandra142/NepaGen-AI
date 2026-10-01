"""Query classification and streaming response generation for NepaGen AI."""
from __future__ import annotations
import re
from typing import Generator

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
from core.retrieval import safe_retrieve


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
    # Only reject if the ENTIRE thing is a URL
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

def classify_query(query: str) -> str:
    """Return a routing category for *query*."""
    compact = normalized_compact_text(query)

    if _is_nonsense_query(compact):
        return "nonsense"
    if compact in {"namaste", "नमस्ते"}:
        return "greeting"
    if compact in GREETING_MAP or compact in CASUAL_SHORT_REPLIES:
        return "casual_chat"
    if compact in {"how are you", "how r you", "how r u", "what's up", "whats up", "k xa", "के छ", "कस्तो छ"}:
        return "casual_chat"
    if compact in {"who are you", "what are you", "what is your name", "what can you do", "tell me about yourself"}:
        return "casual_chat"
    if any(re.match(p, compact) for p in PERSONAL_PATTERNS):
        return "personal_statement"

    words = compact.split()
    casual_terms = ("bye", "night", "morning", "hello", "hi", "namaste", "thanks", "ok", "yes", "no")
    if len(words) <= 3 and any(t in compact for t in casual_terms):
        return "casual_chat"

    # Extended Nepali question words
    nepali_q_starts = (
        "के ", "कसरी ", "किन ", "कहाँ ", "कहिले ",
        "कुन ", "कसले ", "कति ", "कस्तो ",
    )
    factual_terms = (
        "capital", "history", "meaning", "population", "difference",
        "define", "explain", "what is", "who is", "how many",
    )
    question_like = (
        compact.endswith("?")
        or any(compact.startswith(pfx) for pfx in QUESTION_PREFIXES)
        or compact.startswith(nepali_q_starts)
        or any(t in compact for t in factual_terms)
    )

    if question_like:
        rag_terms = ("नेपाल", "nepal", "rag") + factual_terms
        return "RAG_query" if any(t in compact for t in rag_terms) else "factual_question"

    personal_terms = ("i am", "i'm", "my name is", "i like", "i love", "i want", "i study", "i work")
    if any(t in compact for t in personal_terms):
        return "personal_statement"

    return "casual_chat"


# ─── Prompt Builders ──────────────────────────────────────────────────────────

def _conversational_prompt(query: str) -> str:
    profile = st.session_state.get("user_profile", {})
    known_name = profile.get("name")
    ctx = recent_chat_context()
    return (
        f"{CONVERSATIONAL_SYSTEM_PROMPT}\n\n"
        f"Known user info: {f'Name: {known_name}' if known_name else 'None'}\n\n"
        f"Recent conversation:\n{ctx if ctx else 'No recent conversation context.'}\n\n"
        f"User message:\n{query}\n\n"
        f"Assistant reply:"
    )


def _rag_prompt(query: str, context: str) -> str:
    ctx = recent_chat_context()
    return (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"--- UNTRUSTED RETRIEVED CONTEXT START ---\n"
        f"{context}\n"
        f"--- UNTRUSTED RETRIEVED CONTEXT END ---\n\n"
        f"Recent conversation:\n{ctx if ctx else 'No recent conversation context.'}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"Correct Answer in Nepali (no HTML/CSS/code):"
    )


def _prepare_context(query: str, retriever) -> str:
    """Retrieve, filter, and rerank documents for the query."""
    docs = safe_retrieve(retriever, query)
    
    # 1. Sanitize FIRST so noise filtering applies to clean text
    for d in docs:
        d.page_content = _sanitize_html(d.page_content)
        
    # 2. Filter noise conservatively
    docs = [d for d in docs if not _is_noise_doc(d.page_content)]

    # 3. Safe reranking that preserves FAISS similarity but boosts lexical matches
    query_terms = {
        t for t in re.findall(r"[\w\u0900-\u097F]+", normalized_compact_text(query))
        if len(t) > 1
    }

    scored_docs = []
    for idx, doc in enumerate(docs):
        # Base score is the original FAISS rank (0 to N)
        base_score = float(idx) 
        
        # Give a small boost (-0.5 per matching keyword)
        ct = {t for t in re.findall(r"[\w\u0900-\u097F]+", doc.page_content.lower()) if len(t) > 1}
        overlap = len(query_terms & ct)
        
        # Quality score from metadata
        q_score = float(doc.metadata.get("quality_score", 0))
        
        final_score = base_score - (overlap * 0.5) - (q_score * 0.5)
        scored_docs.append((final_score, doc))

    # Sort by the new hybrid score (lower is better)
    scored_docs.sort(key=lambda x: x[0])
    
    # Take top 8
    top_docs = [d for score, d in scored_docs[:8]]
    return "\n\n".join(d.page_content for d in top_docs)


# ─── Streaming Response Generator ─────────────────────────────────────────────

def stream_reply(
    query: str,
    category: str,
    retriever,
    llm,
) -> Generator[str, None, None]:
    """Yield response text chunks. Replaces the old generate_hybrid_reply()."""

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
        context = _prepare_context(query, retriever)
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
