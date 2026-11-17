"""Responsive sidebar navigation for NepaGen AI."""
from __future__ import annotations

import streamlit as st

from core.chat_state import start_new_chat, set_active_chat


def render_sidebar(vector_count: int = 0) -> None:
    with st.sidebar:
        st.markdown("### NepaGen AI")
        st.caption("Grounded Nepali AI")

        st.button(
            "New chat",
            icon=":material/add:",
            key="sidebar-new-chat",
            width="stretch",
            on_click=start_new_chat,
        )

        st.caption("Conversations")
        active_id = st.session_state.active_chat_id
        for conversation in reversed(st.session_state.conversations):
            st.button(
                conversation["title"],
                key=f"chat-nav-{conversation['id']}",
                icon=":material/chat_bubble_outline:",
                width="stretch",
                type="primary" if conversation["id"] == active_id else "secondary",
                on_click=set_active_chat,
                args=(conversation["id"],),
            )

        if vector_count:
            st.caption(f"{vector_count:,} indexed knowledge passages")

        with st.expander("What you can ask", icon=":material/lightbulb:"):
            st.markdown(
                "- **History and geography:** नेपालको इतिहास, राजधानी, भूगोल\n"
                "- **Culture and festivals:** नेपालका चाडपर्व, संस्कृति\n"
                "- **General chat:** नमस्ते, तपाईंलाई कस्तो छ?\n\n"
                "Answers stay grounded in the available Nepali text. "
                "If a fact is unavailable, the app says so instead of guessing."
            )
