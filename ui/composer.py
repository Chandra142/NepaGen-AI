"""Composer (chat input) rendering for NepaGen AI."""
import streamlit as st


def render_composer() -> str | None:
    """Render the mobile-safe chat input and return submitted text, if any."""
    text = st.chat_input(
        "नेपाली वा English मा प्रश्न सोध्नुहोस्…",
        key="composer_text",
        max_chars=2000,
        submit_mode="disable",
    )
    if text and text.strip():
        return text.strip()
    return None
