"""Sidebar rendering for NepaGen AI."""
from __future__ import annotations

import streamlit as st

from core.chat_state import start_new_chat, set_active_chat


def render_sidebar(logo_data_uri: str | None) -> None:
    with st.sidebar:
        # ── Brand block ───────────────────────────────────────────────────────
        if logo_data_uri:
            st.markdown(
                f"""
                <div class="sidebar-brand">
                    <img class="sidebar-logo" src="{logo_data_uri}" alt="NepaGen AI logo" />
                    <div>
                        <div class="sidebar-title">NepaGen AI</div>
                        <div class="sidebar-subtitle">Grounded Nepali AI</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="sidebar-brand">
                    <div style="font-size:1.6rem;line-height:1;">🇳🇵</div>
                    <div>
                        <div class="sidebar-title">NepaGen AI</div>
                        <div class="sidebar-subtitle">Grounded Nepali AI</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── New Chat button ───────────────────────────────────────────────────
        st.markdown('<div class="new-chat-container">', unsafe_allow_html=True)
        if st.button("＋ New Chat", use_container_width=True, key="sidebar-new-chat", type="secondary"):
            start_new_chat()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Chat history list ─────────────────────────────────────────────────
        st.markdown('<div class="sidebar-label">TODAY</div>', unsafe_allow_html=True)
        active_id = st.session_state.active_chat_id
        for conv in reversed(st.session_state.conversations):
            is_active = conv["id"] == active_id
            label = f"💬 {conv['title']}"
            if st.button(label, key=f"chat-nav-{conv['id']}", use_container_width=True, type="primary" if is_active else "secondary"):
                set_active_chat(conv["id"])
                st.rerun()

        # ── What you can ask ──────────────────────────────────────────────────
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">What you can ask</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-card">'
            '<div class="sidebar-subtitle">'
            "<ul style='padding-left: 1.2rem; margin: 0;'>"
            "<li><strong>Nepali History & Geography:</strong> e.g., नेपालको इतिहास, राजधानी, भूगोल</li>"
            "<li><strong>Culture & Traditions:</strong> e.g., नेपालका चाडपर्व, संस्कृति</li>"
            "<li><strong>General Chat:</strong> e.g., नमस्ते, तपाईंलाई कस्तो छ?</li>"
            "</ul>"
            "</div></div>",
            unsafe_allow_html=True,
        )
