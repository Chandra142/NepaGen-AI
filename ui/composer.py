"""Composer (chat input) rendering for NepaGen AI."""
import streamlit as st


def render_composer() -> str | None:
    """Render the sticky chat-input bar and return the submitted text, or None."""
    text = st.chat_input("नेपालीमा प्रश्न सोध्नुस्...", key="composer_text")
    if text and text.strip():
        return text.strip()
    return None
