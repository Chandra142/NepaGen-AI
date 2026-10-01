"""NepaGen AI — Streamlit entry point.

This file is intentionally slim (~100 lines).
All business logic lives in core/, all rendering lives in ui/.

Run:
    streamlit run app.py
"""
import base64
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from vectorstore_store import VectorstoreCompatibilityError, load_vectorstore
from core.retrieval import make_retriever
from core.config import APP_REVISION
from core.chat_state import (
    init_chat_state,
    get_active_chat,
    sync_active_messages,
    build_chat_title,
)
from core.router import classify_query, stream_reply
from core.memory import remember_personal_statement
from ui.styles import inject_styles
from ui.sidebar import render_sidebar
from ui.header import render_brand_header, render_empty_state
from ui.messages import render_user_message, render_assistant_message
from ui.composer import render_composer

# ─── Bootstrap ────────────────────────────────────────────────────────────────

STARTUP_START = time.perf_counter()
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "np rag logo.webp"


def _load_logo_data_uri() -> str | None:
    if not LOGO_PATH.exists():
        return None
    try:
        encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        return f"data:image/webp;base64,{encoded}"
    except Exception as exc:
        print("[startup] failed to load logo:", repr(exc))
        return None


st.set_page_config(page_title="NepaGen AI", page_icon="🇳🇵", layout="wide")
inject_styles()

LOGO_DATA_URI = _load_logo_data_uri()


# ─── Model Loading (cached) ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    from langchain_groq import ChatGroq

    t0 = time.perf_counter()
    db = load_vectorstore()
    retriever = make_retriever(db)
    t1 = time.perf_counter()

    # Support both Streamlit Cloud Secrets and local .env
    # FIX: Streamlit Secrets support added
    groq_key: str | None = None
    try:
        groq_key = st.secrets.get("GROQ_API_KEY")  # type: ignore[arg-type]
    except Exception:
        pass
    if not groq_key:
        groq_key = os.getenv("GROQ_API_KEY")

    # FIX: max_tokens raised 200 → 400 for fuller answers
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=400,
        api_key=groq_key,
    )

    t2 = time.perf_counter()
    print(f"[startup] vectorstore: {t1-t0:.2f}s | llm: {t2-t1:.2f}s")
    print(f"[startup] app revision: {APP_REVISION}")
    return retriever, llm


try:
    retriever, llm = load_models()
except VectorstoreCompatibilityError as exc:
    st.error(str(exc))
    st.warning("Please rebuild the vectorstore with the current embedding model and redeploy.")
    st.stop()
except Exception as exc:
    if "api_key" in str(exc).lower() or "groq" in str(exc).lower():
        st.error("Missing or invalid Groq API key. Please add GROQ_API_KEY to your .env file.")
    else:
        st.error(
            "NepaGen AI could not load the application correctly. "
            "If this is a vectorstore issue, rebuild it with `python src/ingest.py`."
        )
    st.exception(exc)
    st.stop()

print(f"[startup] total: {time.perf_counter() - STARTUP_START:.2f}s")


# ─── Layout ───────────────────────────────────────────────────────────────────

init_chat_state()
render_sidebar(LOGO_DATA_URI)
messages = sync_active_messages()
active_chat = get_active_chat()

render_brand_header(LOGO_DATA_URI, bool(messages), active_chat["title"])

if not messages:
    render_empty_state()

for idx, msg in enumerate(messages):
    if msg["role"] == "assistant":
        render_assistant_message(msg, idx, LOGO_DATA_URI)
    else:
        render_user_message(msg, idx)

# ─── Query Handling (with streaming) ─────────────────────────────────────────

query = render_composer()
pending = st.session_state.pop("pending_query", None)
if not query and pending:
    query = pending

if query:

    # Auto-title the conversation from the first message
    if active_chat["title"] == "New Chat":
        active_chat["title"] = build_chat_title(query)
        st.session_state.chat_titles = [c["title"] for c in st.session_state.conversations]

    # Store user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Classify intent (and remember personal info if needed)
    category = classify_query(query)
    print(f"[router] query category: {category}")
    if category == "personal_statement":
        remember_personal_statement(query)

    # ── Stream the assistant reply ────────────────────────────────────────────
    full_response = ""
    with st.chat_message("assistant", avatar=LOGO_DATA_URI):
        stream_placeholder = st.empty()
        for chunk in stream_reply(query, category, retriever, llm):
            full_response += chunk
            stream_placeholder.markdown(full_response + "▌")
        
        # Clear streaming UI and store the final response
        stream_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()
