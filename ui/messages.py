"""Message rendering for NepaGen AI."""
from __future__ import annotations
import html as html_lib
import json

import streamlit as st
import streamlit.components.v1 as components


# ─── User Message ─────────────────────────────────────────────────────────────

def render_user_message(message: dict, message_index: int) -> None:  # noqa: ARG001
    with st.chat_message("user", avatar="👤"):
        st.markdown(message["content"])

# ─── Assistant Message ────────────────────────────────────────────────────────

def render_assistant_message(message: dict, message_index: int, logo_data_uri: str | None = None) -> None:  # noqa: ARG001
    with st.chat_message("assistant", avatar=logo_data_uri):
        st.markdown(message["content"])


