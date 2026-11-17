"""User profile memory and recent conversation context for NepaGen AI."""
from __future__ import annotations
import re

import streamlit as st


# ─── Text Normalisation ───────────────────────────────────────────────────────

def normalize_query_text(query: str) -> str:
    return query.lower().strip().replace("\\", "").replace("?", "")


def normalize_keyword_text(query: str) -> str:
    return re.sub(r"[^\w\s\u0900-\u097F]", "", query).strip()


def normalized_compact_text(query: str) -> str:
    return normalize_keyword_text(normalize_query_text(query)).lower()


# ─── Profile Memory ───────────────────────────────────────────────────────────

# Only explicit "my name is ..." statements are treated as names. Patterns
# such as "i am ..." describe the user ("i am a student") and must not be
# stored as their name.
_ENGLISH_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^my name is\s+(.+)$"),
    re.compile(r"^i am called\s+(.+)$"),
    re.compile(r"^call me\s+(.+)$"),
)
_NEPALI_NAME_PATTERN = re.compile(r"^मेरो नाम\s+(.+?)(?:\s+(?:हो|हुँ|छु))?\s*$")

# Leading determiners signal a description, not a name.
_NAME_STOPWORDS = {"a", "an", "the", "what", "who", "के", "कुन", "कस", "कहाँ", "कहिले"}


def _clean_name(raw_name: str) -> str | None:
    name = raw_name.strip(" .!?,।")
    tokens = name.split()
    if not name or len(name) > 40 or not 1 <= len(tokens) <= 4:
        return None
    if tokens[0].lower() in _NAME_STOPWORDS:
        return None
    return name if re.search(r"[ऀ-ॿ]", name) else name.title()


def extract_person_name(query: str) -> str | None:
    """Return an explicitly stated name, or None when the text is not one."""
    compact = normalized_compact_text(query)

    for pattern in _ENGLISH_NAME_PATTERNS:
        match = pattern.match(compact)
        if match:
            return _clean_name(match.group(1))

    match = _NEPALI_NAME_PATTERN.match(compact)
    if match:
        return _clean_name(match.group(1))

    return None


def remember_personal_statement(query: str) -> None:
    """Persist explicitly stated profile facts from a personal statement."""
    profile = st.session_state.setdefault("user_profile", {})
    name = extract_person_name(query)
    if name:
        profile["name"] = name


def recent_chat_context(limit: int = 4, exclude_latest: bool = False) -> str:
    """Return recent messages, optionally excluding the current user query."""
    messages = st.session_state.get("messages", [])
    if exclude_latest and messages and messages[-1].get("role") == "user":
        messages = messages[:-1]
    recent = messages[-limit:]
    if not recent:
        return ""
    lines = []
    for item in recent:
        role = "User" if item["role"] == "user" else "Assistant"
        lines.append(f"{role}: {item['content']}")
    return "\n".join(lines)
