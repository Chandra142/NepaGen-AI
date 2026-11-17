"""Brand header and responsive empty-state rendering for NepaGen AI."""
from __future__ import annotations

import streamlit as st

from core.chat_state import queue_prompt


def render_brand_header(
    logo_data_uri: str | None,
    has_messages: bool,
    chat_title: str,
    rag_ready: bool = True,
) -> None:
    if has_messages:
        with st.container(key="chat-header", horizontal=True, vertical_alignment="center"):
            st.markdown("**NepaGen AI**")
            st.space("stretch")
            st.badge(
                "RAG ready" if rag_ready else "Index unavailable",
                icon=":material/check_circle:" if rag_ready else ":material/error:",
                color="green" if rag_ready else "orange",
            )
        st.caption(chat_title)
        return

    with st.container(key="hero", horizontal_alignment="center"):
        if logo_data_uri:
            st.image(logo_data_uri, width=84, alt="NepaGen AI logo")
        else:
            st.markdown("🇳🇵", text_alignment="center")
        st.title("NepaGen AI", text_alignment="center")
        st.markdown("**नेपालीमा सोध्नुहोस्। ज्ञानसँग जोडिनुहोस्।**", text_alignment="center")
        st.caption(
            "Grounded answers in Nepali · Ask in Nepali or English",
            text_alignment="center",
        )


def render_empty_state() -> None:
    st.markdown("#### सुरु गर्नका लागि केही प्रश्नहरू", text_alignment="center")
    st.caption("Tap a suggestion or type your own question below.", text_alignment="center")

    prompt_chips = [
        ("सगरमाथाको बारेमा बताउनुहोस्।", ":material/landscape:"),
        ("नेपालको संस्कृति के हो?", ":material/account_balance:"),
        ("नेपालको इतिहासबारे बताउनुहोस्।", ":material/history_edu:"),
        ("नेपालका प्रमुख पर्वहरू के के हुन्?", ":material/celebration:"),
    ]
    with st.grid("auto", min_column_width=220, gap="small"):
        for index, (prompt, icon) in enumerate(prompt_chips):
            st.button(
                prompt,
                key=f"prompt-chip-{index}",
                icon=icon,
                width="stretch",
                wrap=True,
                on_click=queue_prompt,
                args=(prompt,),
            )
