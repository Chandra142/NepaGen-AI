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

def remember_personal_statement(query: str) -> None:
    """Parse a personal statement and persist any extractable info to the user profile."""
    profile = st.session_state.setdefault("user_profile", {})
    compact = normalized_compact_text(query)

    en_match = re.match(r"^(?:i am|i'm|my name is)\s+(.+)$", compact, flags=re.IGNORECASE)
    np_match = re.match(r"^(?:मेरो नाम|म)\s+(.+?)(?:\s+(?:हो|हुँ|छु))?\s*$", compact)

    if en_match:
        name = en_match.group(1).strip(" .!?,")
        if name:
            profile["name"] = name.title()
            return

    if np_match:
        name = np_match.group(1).strip(" .!?,")
        if name:
            profile["name"] = name


def recent_chat_context(limit: int = 4) -> str:
    """Return a formatted string of the most recent messages for prompt injection."""
    recent = st.session_state.get("messages", [])[-limit:]
    if not recent:
        return ""
    lines = []
    for item in recent:
        role = "User" if item["role"] == "user" else "Assistant"
        lines.append(f"{role}: {item['content']}")
    return "\n".join(lines)
