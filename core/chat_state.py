"""Chat session-state management for NepaGen AI."""
from __future__ import annotations
import re

import streamlit as st


def init_chat_state() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = [
            {"id": "chat-1", "title": "New Chat", "messages": []}
        ]
        st.session_state.active_chat_id = "chat-1"
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}
    _sync_titles()


def get_active_chat() -> dict:
    active_id = st.session_state.get("active_chat_id")
    for conv in st.session_state.conversations:
        if conv["id"] == active_id:
            return conv
    # Fallback: use first conversation
    st.session_state.active_chat_id = st.session_state.conversations[0]["id"]
    return st.session_state.conversations[0]


def set_active_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id


def start_new_chat() -> None:
    chat_number = len(st.session_state.conversations) + 1
    new_chat = {
        "id": f"chat-{chat_number}",
        "title": "New Chat",
        "messages": [],
    }
    st.session_state.conversations.append(new_chat)
    st.session_state.active_chat_id = new_chat["id"]
    _sync_titles()


def sync_active_messages() -> list:
    active_chat = get_active_chat()
    st.session_state.messages = active_chat["messages"]
    _sync_titles()
    return active_chat["messages"]


def build_chat_title(text: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    return (clean[:32] + ("…" if len(clean) > 32 else "")) if clean else "New Chat"


def queue_prompt(prompt: str) -> None:
    st.session_state.pending_query = prompt


def _sync_titles() -> None:
    st.session_state.chat_titles = [c["title"] for c in st.session_state.conversations]
