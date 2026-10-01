"""Brand header and empty state rendering for NepaGen AI."""
from __future__ import annotations

import streamlit as st

from core.chat_state import queue_prompt


def render_brand_header(logo_data_uri: str | None, has_messages: bool, chat_title: str) -> None:
    if has_messages:
        logo_html = f'<img class="top-nav-logo" src="{logo_data_uri}" alt="Logo" />' if logo_data_uri else '🇳🇵'
        st.markdown(
            f"""
            <div class="top-nav">
                <div class="top-nav-left">
                    {logo_html}
                    <div class="top-nav-title">NepaGen AI</div>
                </div>
                <div class="top-nav-center">{chat_title}</div>
                <div class="top-nav-right">
                    <div class="rag-indicator"></div>
                    <span>RAG Ready</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        logo_html = f'<img class="hero-logo" src="{logo_data_uri}" alt="NepaGen AI logo" />' if logo_data_uri else '<div style="font-size:4rem; margin-bottom: 1rem;">🇳🇵</div>'
        st.markdown(
            f"""
            <div class="hero-container">
                {logo_html}
                <h1 class="hero-title">NepaGen AI</h1>
                <div class="hero-subtitle">Grounded Nepali AI</div>
                <p class="hero-tagline">नेपालीमा सोध्नुहोस्। ज्ञानसँग जोडिनुहोस्।</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_empty_state() -> None:
    prompt_chips = [
        ("🏔️", "सगरमाथाको बारेमा बताउनुहोस्।"),
        ("🏛️", "नेपालको संस्कृति के हो?"),
        ("📜", "नेपालको इतिहासबारे बताउनुहोस्।"),
        ("🇳🇵", "नेपालका प्रमुख पर्वहरू के के हुन्?")
    ]
    
    # We will use st.columns for the grid to keep interactivity
    st.markdown('<div class="prompt-grid">', unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, (icon, prompt) in enumerate(prompt_chips):
        with cols[idx % 2]:
            st.markdown('<div class="prompt-card-wrapper">', unsafe_allow_html=True)
            if st.button(f"{icon} {prompt}", key=f"prompt-chip-{idx}", use_container_width=True):
                queue_prompt(prompt)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
