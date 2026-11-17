"""Chat message and retrieval-reference rendering for NepaGen AI."""
from __future__ import annotations

import streamlit as st


def render_sources(sources: list) -> None:
    """Render compact, expandable excerpts supporting an answer."""
    if not sources:
        return
    with st.expander(f"References ({len(sources)})", icon=":material/menu_book:"):
        for index, document in enumerate(sources, start=1):
            excerpt = " ".join(document.page_content.split())
            if len(excerpt) > 450:
                excerpt = excerpt[:450].rstrip() + "…"
            st.markdown(f"**Reference {index}**")
            st.caption(excerpt)


def render_user_message(message: dict, message_index: int) -> None:
    with st.container(key=f"user-message-{message_index}", horizontal_alignment="right"):
        with st.chat_message("user", avatar=":material/person:", width="content"):
            st.markdown(message["content"])


def render_assistant_message(
    message: dict,
    message_index: int,  # noqa: ARG001
    logo_path: str | None = None,
) -> None:
    with st.chat_message(
        "assistant",
        avatar=logo_path or ":material/smart_toy:",
        width="stretch",
    ):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))
