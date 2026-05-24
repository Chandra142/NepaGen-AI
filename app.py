import base64
import html as html_lib
import json
import re
import time
from pathlib import Path
STARTUP_START = time.perf_counter()
APP_REVISION = "2026-05-24-safe-retrieval"


import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from vectorstore_store import VectorstoreCompatibilityError, load_vectorstore

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "np rag logo.webp"


def load_logo_data_uri() -> str | None:
    if not LOGO_PATH.exists():
        return None
    try:
        logo_bytes = LOGO_PATH.read_bytes()
        encoded = base64.b64encode(logo_bytes).decode("ascii")
        return f"data:image/webp;base64,{encoded}"
    except Exception as exc:
        print("[startup] failed to load logo:", repr(exc))
        return None


LOGO_DATA_URI = load_logo_data_uri()

st.set_page_config(page_title="NepaGen AI", page_icon="🇳🇵", layout="wide")

st.markdown(
    """
    <style>
    :root {
        color-scheme: light dark;
        --bg: #0b1020;
        --bg-2: #111827;
        --panel: rgba(17, 24, 39, 0.74);
        --panel-strong: rgba(15, 23, 42, 0.92);
        --panel-soft: rgba(31, 41, 55, 0.62);
        --text: #eef2ff;
        --muted: #94a3b8;
        --muted-2: #cbd5e1;
        --border: rgba(255, 255, 255, 0.08);
        --border-strong: rgba(255, 255, 255, 0.14);
        --crimson: #b91c1c;
        --crimson-2: #7f1d1d;
        --gold: #d97706;
        --gold-soft: rgba(217, 119, 6, 0.16);
        --shadow-lg: 0 20px 60px rgba(2, 6, 23, 0.48);
        --shadow-md: 0 16px 34px rgba(2, 6, 23, 0.28);
        --shadow-sm: 0 10px 20px rgba(2, 6, 23, 0.22);
    }
    @media (prefers-color-scheme: light) {
        :root {
            color-scheme: light;
            --bg: #f4f7fb;
            --bg-2: #edf2f8;
            --panel: rgba(250, 252, 255, 0.9);
            --panel-strong: rgba(247, 250, 253, 0.96);
            --panel-soft: rgba(234, 240, 247, 0.92);
            --text: #132033;
            --muted: #536273;
            --muted-2: #3b4a5b;
            --border: rgba(15, 23, 42, 0.08);
            --border-strong: rgba(15, 23, 42, 0.14);
            --shadow-lg: 0 18px 44px rgba(15, 23, 42, 0.08);
            --shadow-md: 0 14px 24px rgba(15, 23, 42, 0.07);
            --shadow-sm: 0 8px 14px rgba(15, 23, 42, 0.05);
        }
    }
    * { box-sizing: border-box; }
    html, body {
        width: 100%;
        min-height: 100%;
        margin: 0;
        overflow-x: hidden;
        background:
            radial-gradient(circle at 12% 0%, rgba(185, 28, 28, 0.18), transparent 26%),
            radial-gradient(circle at 88% 12%, rgba(217, 119, 6, 0.14), transparent 22%),
            linear-gradient(180deg, #050816 0%, #0b1020 42%, #111827 100%);
        color: var(--text);
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif, "Noto Sans Devanagari";
        -webkit-text-size-adjust: 100%;
        text-size-adjust: 100%;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    .stApp, .main {
        background:
            radial-gradient(circle at 12% 0%, rgba(185, 28, 28, 0.18), transparent 26%),
            radial-gradient(circle at 88% 12%, rgba(217, 119, 6, 0.14), transparent 22%),
            linear-gradient(180deg, #050816 0%, #0b1020 42%, #111827 100%) !important;
        color: var(--text) !important;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
            radial-gradient(circle at 20% 18%, rgba(255, 255, 255, 0.04), transparent 18%),
            radial-gradient(circle at 80% 12%, rgba(255, 255, 255, 0.03), transparent 15%),
            linear-gradient(180deg, rgba(255, 255, 255, 0.015), transparent 14%, transparent 86%, rgba(0, 0, 0, 0.18));
    }
    .block-container {
        max-width: 100%;
        padding-top: 0.8rem;
        padding-bottom: 6rem;
        padding-left: clamp(0.6rem, 1.7vw, 1.25rem);
        padding-right: clamp(0.6rem, 1.7vw, 1.25rem);
        margin: 0 auto;
    }
    @media (prefers-color-scheme: light) {
        html, body {
            background:
                radial-gradient(circle at 14% 0%, rgba(185, 28, 28, 0.05), transparent 24%),
                radial-gradient(circle at 88% 10%, rgba(217, 119, 6, 0.05), transparent 20%),
                linear-gradient(180deg, #fbfcfe 0%, #f4f7fb 48%, #edf2f8 100%);
            color: var(--text);
        }
        .stApp, .main {
            background:
                radial-gradient(circle at 14% 0%, rgba(185, 28, 28, 0.05), transparent 24%),
                radial-gradient(circle at 88% 10%, rgba(217, 119, 6, 0.05), transparent 20%),
                linear-gradient(180deg, #fbfcfe 0%, #f4f7fb 48%, #edf2f8 100%) !important;
            color: var(--text) !important;
        }
        .stApp::before {
            background:
                radial-gradient(circle at 20% 18%, rgba(15, 23, 42, 0.02), transparent 18%),
                radial-gradient(circle at 80% 12%, rgba(15, 23, 42, 0.018), transparent 15%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.12), transparent 14%, transparent 86%, rgba(15, 23, 42, 0.035));
        }
        section[data-testid="stSidebar"],
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(250, 252, 255, 0.98), rgba(243, 247, 252, 0.98));
            border-right: 1px solid rgba(15, 23, 42, 0.06);
            box-shadow: inset -1px 0 0 rgba(15, 23, 42, 0.03);
        }
        [data-testid="stSidebar"] button,
        .sidebar-brand, .sidebar-card,
        .brand-shell,
        .composer-shell,
        .message-card--assistant,
        .message-card--user {
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
        }
        [data-testid="stSidebar"] button {
            background: linear-gradient(180deg, rgba(250, 252, 255, 1), rgba(240, 245, 251, 1)) !important;
            border-color: rgba(15, 23, 42, 0.08) !important;
            color: var(--text) !important;
        }
        .sidebar-brand, .sidebar-card {
            background: linear-gradient(180deg, rgba(250, 252, 255, 0.98), rgba(241, 246, 251, 0.96));
            border-color: rgba(15, 23, 42, 0.08);
        }
        .brand-shell,
        .composer-shell {
            background: linear-gradient(180deg, rgba(250, 252, 255, 0.98), rgba(242, 247, 252, 0.96));
            border-color: rgba(15, 23, 42, 0.08);
        }
        .brand-tagline,
        .brand-subcopy,
        .sidebar-subtitle,
        .composer-label,
        .hero-copy {
            color: #475569;
        }
        .message-card--assistant {
            background: linear-gradient(180deg, rgba(251, 252, 255, 0.98), rgba(244, 248, 253, 0.98));
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-left: 4px solid rgba(217, 119, 6, 0.85);
        }
        .message-card--assistant::before {
            background: linear-gradient(135deg, rgba(217, 119, 6, 0.05), transparent 42%);
        }
        .message-card--user {
            background: linear-gradient(180deg, rgba(255, 247, 247, 0.98), rgba(251, 241, 241, 0.96));
            border: 1px solid rgba(185, 28, 28, 0.12);
        }
        .message-content {
            color: var(--text) !important;
        }
        .message-copy-button {
            background: rgba(246, 249, 252, 0.98) !important;
            border-color: rgba(15, 23, 42, 0.08) !important;
            color: #0f172a !important;
        }
        .composer-wrap {
            background: linear-gradient(180deg, rgba(244, 247, 251, 0), rgba(244, 247, 251, 0.7) 18%, rgba(244, 247, 251, 0.96));
        }
        div[data-testid="stChatInput"] {
            background: linear-gradient(180deg, rgba(244, 247, 251, 0.02), rgba(244, 247, 251, 0.94));
            border-top-color: rgba(15, 23, 42, 0.08);
        }
        div[data-testid="stChatInput"] textarea,
        div[data-testid="stChatInput"] input,
        .stChatInput textarea,
        .stChatInput input {
            background: linear-gradient(180deg, rgba(250, 252, 255, 0.98), rgba(243, 247, 252, 0.98)) !important;
            color: var(--text) !important;
            border-color: rgba(15, 23, 42, 0.1) !important;
            box-shadow: 0 10px 18px rgba(15, 23, 42, 0.06);
        }
    }
    [data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"] {
        width: 17rem !important;
        min-width: 17rem !important;
        max-width: 17rem !important;
        background: linear-gradient(180deg, rgba(10, 14, 26, 0.94), rgba(13, 19, 34, 0.92));
        border-right: 1px solid rgba(255, 255, 255, 0.04);
        box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.03);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 0.9rem; padding-bottom: 1rem; }
    [data-testid="stSidebar"] button {
        width: 100%;
        border-radius: 0.95rem;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        background: linear-gradient(180deg, rgba(16, 22, 38, 0.86), rgba(14, 20, 34, 0.82)) !important;
        color: var(--text) !important;
        box-shadow: 0 8px 16px rgba(2, 6, 23, 0.12);
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    }
    [data-testid="stSidebar"] button:hover {
        border-color: rgba(217, 119, 6, 0.28) !important;
        box-shadow: 0 12px 20px rgba(2, 6, 23, 0.18);
        transform: translateY(-1px);
    }
    .sidebar-brand, .sidebar-card {
        border: 1px solid rgba(255, 255, 255, 0.06);
        background: linear-gradient(180deg, rgba(16, 22, 38, 0.9), rgba(14, 20, 34, 0.76));
        border-radius: 1rem;
        padding: 0.9rem;
        box-shadow: 0 10px 20px rgba(2, 6, 23, 0.16);
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .sidebar-logo {
        width: 2.6rem;
        height: 2.6rem;
        border-radius: 18px;
        object-fit: cover;
        box-shadow: 0 10px 24px rgba(2, 6, 23, 0.35), 0 0 0 1px rgba(255, 255, 255, 0.08) inset;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.04);
    }
    .sidebar-brand { margin-bottom: 0.75rem; }
    .sidebar-title { font-size: 1rem; font-weight: 800; margin: 0.1rem 0 0; letter-spacing: -0.01em; }
    .sidebar-subtitle { color: var(--muted); font-size: 0.86rem; margin: 0.15rem 0 0; line-height: 1.5; }
    .brand-header {
        max-width: 760px;
        width: 100%;
        margin: 0 auto 1rem;
        padding: 1rem 1rem 0.4rem;
        display: flex;
        justify-content: center;
    }
    .brand-shell {
        width: 100%;
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.1rem;
        border-radius: 1.5rem;
        background: linear-gradient(180deg, rgba(18, 24, 40, 0.9), rgba(14, 20, 34, 0.76));
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 18px 44px rgba(2, 6, 23, 0.28);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }
    .brand-logo {
        width: 4.1rem;
        height: 4.1rem;
        border-radius: 18px;
        object-fit: cover;
        box-shadow: 0 14px 30px rgba(2, 6, 23, 0.42), 0 0 0 1px rgba(255, 255, 255, 0.08) inset;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.04);
        flex: 0 0 auto;
    }
    .brand-copy { min-width: 0; }
    .brand-title {
        font-size: clamp(1.55rem, 3vw, 2rem);
        font-weight: 860;
        letter-spacing: -0.04em;
        margin: 0;
        line-height: 1.05;
    }
    .brand-tagline {
        margin: 0.35rem 0 0;
        color: var(--muted-2);
        font-size: 0.96rem;
        line-height: 1.55;
    }
    .brand-eyebrow {
        margin: 0 0 0.2rem;
        color: rgba(217, 119, 6, 0.86);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .brand-subcopy {
        margin: 0.25rem 0 0;
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.45;
    }
    .sidebar-label { color: #dbe4f3; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; margin: 0.85rem 0 0.4rem; }
    .sidebar-divider { height: 1px; background: var(--border); margin: 0.85rem 0; }
    .chat-layout {
        max-width: 760px;
        width: 100%;
        margin: 0 auto;
        padding: 2rem 1rem 0;
        display: flex;
        flex-direction: column;
    }
    .messages-wrapper {
        display: flex;
        flex-direction: column;
        width: 100%;
        gap: 0.95rem;
        padding: 0.1rem 0 1rem;
    }
    .message-row {
        display: flex;
        width: 100%;
        margin-bottom: 0.1rem;
        min-width: 0;
        align-items: flex-start;
    }
    .message-row--assistant { justify-content: flex-start; }
    .message-row--user { justify-content: flex-end; }
    .message-card {
        position: relative;
        display: flex;
        flex-direction: column;
        gap: 0.65rem;
        width: fit-content;
        max-width: 720px;
        min-width: 120px;
        padding: 18px 22px;
        border-radius: 22px;
        white-space: pre-wrap;
        word-break: break-word;
        overflow-wrap: break-word;
        height: auto;
        box-sizing: border-box;
        box-shadow: var(--shadow-md);
    }
    .message-card--assistant {
        background: linear-gradient(180deg, rgba(18, 24, 40, 0.98), rgba(14, 20, 34, 0.94));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid rgba(217, 119, 6, 0.82);
        box-shadow:
            0 22px 44px rgba(2, 6, 23, 0.34),
            0 0 0 1px rgba(255, 255, 255, 0.02) inset;
        padding-top: 2.35rem;
    }
    .message-card--assistant::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        background: linear-gradient(135deg, rgba(217, 119, 6, 0.08), transparent 42%);
        pointer-events: none;
    }
    .message-card--user {
        background: linear-gradient(180deg, rgba(120, 30, 30, 0.94), rgba(85, 18, 18, 0.92));
        border: 1px solid rgba(185, 28, 28, 0.32);
        box-shadow: 0 16px 30px rgba(185, 28, 28, 0.12);
    }
    .message-copy-button {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 2;
        width: 2.15rem !important;
        min-width: 2.15rem !important;
        height: 2.15rem !important;
        padding: 0 !important;
        border-radius: 999px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: rgba(15, 23, 42, 0.92) !important;
        color: #e2e8f0 !important;
        box-shadow: 0 10px 16px rgba(2, 6, 23, 0.18);
        line-height: 1 !important;
        font-size: 0.95rem !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        cursor: pointer;
        transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease, box-shadow 0.18s ease;
    }
    .message-copy-button:hover {
        border-color: rgba(217, 119, 6, 0.35) !important;
        transform: translateY(-1px);
        box-shadow: 0 12px 18px rgba(2, 6, 23, 0.24);
    }
    .message-copy-button:active { transform: translateY(0); }
    .message-content {
        min-width: 0;
        width: 100%;
        color: var(--text);
        font-size: 1rem;
        line-height: 1.75;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif, "Noto Sans Devanagari";
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }
    .composer-wrap {
        position: sticky;
        bottom: 0;
        z-index: 30;
        width: 100%;
        padding: 0.95rem 0 0.65rem;
        background: linear-gradient(180deg, rgba(11, 16, 32, 0), rgba(11, 16, 32, 0.66) 18%, rgba(11, 16, 32, 0.95));
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }
    .composer-shell {
        max-width: 760px;
        width: 100%;
        margin: 0 auto;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: linear-gradient(180deg, rgba(18, 24, 40, 0.95), rgba(14, 20, 34, 0.92));
        border-radius: 1.45rem;
        box-shadow: 0 18px 42px rgba(2, 6, 23, 0.38);
        padding: 0.95rem;
    }
    .composer-label {
        color: var(--muted);
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0 0 0.45rem 0.2rem;
    }
    .stChatInput {
        max-width: 760px;
        width: 100%;
        margin: 0 auto;
    }
    div[data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        z-index: 20;
        background: linear-gradient(180deg, rgba(11, 16, 32, 0.02), rgba(11, 16, 32, 0.92));
        padding-top: 0.6rem;
        padding-bottom: max(0.75rem, env(safe-area-inset-bottom));
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }
    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] input,
    .stChatInput textarea,
    .stChatInput input {
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(15, 23, 42, 0.94)) !important;
        color: var(--text) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 1rem !important;
        min-height: 3.25rem !important;
        padding: 0.95rem 1rem !important;
        line-height: 1.55 !important;
        box-shadow: 0 14px 26px rgba(2, 6, 23, 0.22);
    }
    .stChatInput textarea::placeholder, .stChatInput input::placeholder { color: var(--muted) !important; }
    @media (max-width: 48rem) {
        .block-container { padding-left: 0.6rem; padding-right: 0.6rem; padding-bottom: 6rem; }
        .brand-header { padding: 0.5rem 0 0.8rem; margin-bottom: 0.6rem; }
        .brand-shell { padding: 0.85rem; gap: 0.85rem; }
        .brand-logo { width: 3.55rem; height: 3.55rem; }
        .sidebar-brand { gap: 0.65rem; }
        .hero-wrap { padding: 1.05rem 0.85rem 0.85rem; }
        .hero-title { font-size: clamp(1.8rem, 8vw, 2.4rem); }
        .prompt-grid { grid-template-columns: 1fr; }
        .chat-layout { padding: 1.25rem 0.35rem 0; }
        .messages-wrapper { gap: 0.8rem; }
        .message-card { max-width: 92%; padding: 16px 18px; border-radius: 18px; }
        .message-card--assistant { padding-top: 2.2rem; }
        .composer-shell { padding: 0.8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_query_text(query: str) -> str:
    return query.lower().strip().replace("\\", "").replace("?", "")


def normalize_keyword_text(query: str) -> str:
    return re.sub(r"[^\w\s\u0900-\u097F]", "", query).strip()


CONVERSATIONAL_SYSTEM_PROMPT = """
You are NepaGen AI, a warm, natural Nepali-first assistant.

Rules:
- Respond like a helpful human conversation partner.
- Do not sound like search results, datasets, or retrieval fragments.
- Keep casual replies brief, friendly, and natural, usually 1-2 short sentences.
- For personal statements, acknowledge warmly and optionally remember the user's self-introduction.
- If the user speaks in Nepali, prefer Nepali.
- If the user speaks in English, respond in English or a comfortable mix.
""".strip()


RAG_SYSTEM_PROMPT = """
You are NepaGen AI, a factual Nepali QA assistant.

Rules:
- Use ONLY the provided context.
- If the answer is not in the context, say: "मलाई जानकारी भेटिएन।"
- Keep the answer short, direct, and factual.
- Do not mention retrieval, embeddings, or vector stores.
""".strip()


NONSENSE_RESPONSE = "मलाई बुझिएन। कृपया फेरि प्रश्न सोध्नुहोस्।"


GREETING_MAP = {
    "hi": "नमस्ते 👋",
    "hello": "नमस्ते 👋",
    "hey": "नमस्ते 👋",
    "good night": "शुभ रात्री 🌙",
    "good morning": "शुभ प्रभात ☀️",
    "good evening": "शुभ साँझ 🌙",
    "bye": "फेरि भेटौँला 👋",
    "namaste": "नमस्ते 👋",
    "नमस्ते": "नमस्ते 👋",
}

CASUAL_SHORT_REPLIES = {
    "ok": "ठिक छ 👍",
    "okay": "ठिक छ 👍",
    "yes": "हुन्छ 👍",
    "yep": "हुन्छ 👍",
    "no": "हुन्न 👍",
    "nah": "हुन्न 👍",
    "hmm": "सोच्दै छु 🤔",
    "hmmm": "सोच्दै छु 🤔",
    "thanks": "स्वागत छ 😊",
    "thank you": "स्वागत छ 😊",
}

QUESTION_PREFIXES = (
    "what is",
    "who is",
    "where is",
    "when is",
    "why is",
    "how is",
    "what are",
    "who are",
    "where are",
    "when are",
    "why are",
    "how are",
    "define",
    "explain",
    "meaning of",
    "capital of",
    "history of",
    "tell me about",
    "नेपालको",
    "नेपाल",
    "राग",
    "rag",
)

PERSONAL_PATTERNS = (
    r"^i am\s+.+",
    r"^i'm\s+.+",
    r"^my name is\s+.+",
    r"^i like\s+.+",
    r"^i love\s+.+",
    r"^i want\s+.+",
    r"^i study\s+.+",
    r"^i work\s+.+",
    r"^म\s+.+(हुँ|छु|हो)\s*$",
    r"^मेरो नाम\s+.+",
    r"^मलाई\s+.+मन पर्छ.*",
)


def _normalized_compact_text(query: str) -> str:
    return normalize_keyword_text(normalize_query_text(query)).lower()


def _is_nonsense_query(compact_query: str) -> bool:
    if not compact_query:
        return True
    if re.fullmatch(r"[0-9\s]+", compact_query):
        return True
    if re.fullmatch(r"[a-z]+", compact_query) and len(compact_query) >= 4 and not re.search(r"[aeiou]", compact_query):
        return True
    if re.search(r"(.)\1{6,}", compact_query):
        return True
    if len(compact_query) <= 2 and compact_query not in {"ok", "no", "yes", "hmm"}:
        return True
    return False


def classify_query(query: str) -> str:
    compact_query = _normalized_compact_text(query)
    if _is_nonsense_query(compact_query):
        return "nonsense"

    if compact_query in {"namaste", "नमस्ते"}:
        return "greeting"

    if compact_query in GREETING_MAP:
        return "casual_chat"

    if compact_query in CASUAL_SHORT_REPLIES:
        return "casual_chat"

    if compact_query in {"how are you", "how r you", "how r u", "what's up", "whats up", "k xa", "के छ", "कस्तो छ"}:
        return "casual_chat"

    if compact_query in {"who are you", "what are you", "what is your name", "what can you do", "tell me about yourself"}:
        return "casual_chat"

    if any(re.match(pattern, compact_query) for pattern in PERSONAL_PATTERNS):
        return "personal_statement"

    words = compact_query.split()
    if len(words) <= 3 and any(term in compact_query for term in ("bye", "night", "morning", "hello", "hi", "namaste", "thanks", "ok", "yes", "no")):
        return "casual_chat"

    question_like = (
        compact_query.endswith("?")
        or any(compact_query.startswith(prefix) for prefix in QUESTION_PREFIXES)
        or compact_query.startswith(("के ", "कसरी ", "किन ", "कहाँ ", "कहिले ", "कुन ", "कसले "))
        or any(term in compact_query for term in ("capital", "history", "meaning", "population", "difference", "define", "explain", "what is", "who is", "how many"))
    )

    if question_like:
        return "RAG_query" if any(term in compact_query for term in ("नेपाल", "nepal", "rag", "capital", "history", "meaning", "population", "difference", "define", "explain", "what is", "who is", "how many")) else "factual_question"

    if any(term in compact_query for term in ("i am", "i'm", "my name is", "i like", "i love", "i want", "i study", "i work")):
        return "personal_statement"

    if len(words) <= 3:
        return "casual_chat"

    return "casual_chat"


def _remember_personal_statement(query: str) -> None:
    profile = st.session_state.setdefault("user_profile", {})
    compact_query = _normalized_compact_text(query)

    english_name_match = re.match(r"^(?:i am|i'm|my name is)\s+(.+)$", compact_query, flags=re.IGNORECASE)
    nepali_name_match = re.match(r"^(?:मेरो नाम|म)\s+(.+?)(?:\s+(?:हो|हुँ|छु))?\s*$", compact_query)

    if english_name_match:
        name = english_name_match.group(1).strip(" .!?,")
        if name:
            profile["name"] = name.title()
            return

    if nepali_name_match:
        name = nepali_name_match.group(1).strip(" .!?,")
        if name:
            profile["name"] = name


def _recent_chat_context(limit: int = 4) -> str:
    recent_messages = st.session_state.get("messages", [])[-limit:]
    if not recent_messages:
        return ""
    formatted = []
    for item in recent_messages:
        role = "User" if item["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {item['content']}")
    return "\n".join(formatted)


def _generate_conversational_reply(query: str, category: str) -> str:
    compact_query = _normalized_compact_text(query)
    if compact_query in GREETING_MAP:
        return GREETING_MAP[compact_query]
    if compact_query in CASUAL_SHORT_REPLIES:
        return CASUAL_SHORT_REPLIES[compact_query]
    if category == "greeting" and compact_query in {"how are you", "how r you", "how r u", "what's up", "whats up", "k xa", "के छ", "कस्तो छ"}:
        return "म राम्रो छु 😊 तपाईंलाई कसरी सहयोग गरौं?"

    profile = st.session_state.get("user_profile", {})
    known_name = profile.get("name")
    recent_context = _recent_chat_context()

    prompt = f"""
{CONVERSATIONAL_SYSTEM_PROMPT}

Known user info: {f'Name: {known_name}' if known_name else 'None'}

Recent conversation:
{recent_context if recent_context else 'No recent conversation context.'}

User message:
{query}

Assistant reply:
"""

    response = llm.invoke(prompt)
    reply = (response.content or "").strip()
    if not reply:
        return "म राम्रो छु 😊 तपाईंलाई कसरी सहयोग गरौं?"
    return reply


def generate_hybrid_reply(query: str, active_chat: dict) -> tuple[str, str]:
    category = classify_query(query)
    print("[router] category:", category)

    if category == "nonsense":
        return category, NONSENSE_RESPONSE

    if category == "personal_statement":
        _remember_personal_statement(query)
        return category, _generate_conversational_reply(query, category)

    if category in {"greeting", "casual_chat"}:
        return category, _generate_conversational_reply(query, category)

    if category in {"factual_question", "RAG_query"}:
        docs = safe_retrieve_documents(retriever, query)
        docs_sorted = sorted(docs, key=lambda d: (-float(d.metadata.get("quality_score", 0)), -len(d.page_content)))
        filtered_docs = [d for d in docs_sorted if not _is_noise_text(d.page_content)]
        query_terms = {term for term in re.findall(r"[\w\u0900-\u097F]+", _normalized_compact_text(query)) if len(term) > 1}

        def _rerank(doc):
            content = normalize_query_text(doc.page_content.lower())
            content_terms = {term for term in re.findall(r"[\w\u0900-\u097F]+", content) if len(term) > 1}
            overlap = len(query_terms & content_terms)
            quality = float(doc.metadata.get("quality_score", 0))
            return (-overlap, -quality, -len(doc.page_content))

        filtered_docs = sorted(filtered_docs, key=_rerank)
        top_docs = filtered_docs[:8]
        context = "\n\n".join([d.page_content for d in top_docs])

        if not context.strip():
            return category, "मलाई जानकारी भेटिएन। कृपया प्रश्नलाई अलि फरक तरिकाले सोध्नुहोस्।"

        prompt = f"""
{RAG_SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Correct Answer in Nepali:
"""

        response = llm.invoke(prompt)
        answer = (response.content or "").strip()
        if not answer:
            return category, "मलाई जानकारी भेटिएन।"
        return category, answer

    return category, _generate_conversational_reply(query, "casual_chat")


def init_chat_state() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = [{"id": "chat-1", "title": "New Chat", "messages": []}]
        st.session_state.active_chat_id = "chat-1"
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}
    st.session_state.chat_titles = [conversation["title"] for conversation in st.session_state.conversations]


def get_active_chat() -> dict:
    active_chat_id = st.session_state.get("active_chat_id")
    for conversation in st.session_state.conversations:
        if conversation["id"] == active_chat_id:
            return conversation
    st.session_state.active_chat_id = st.session_state.conversations[0]["id"]
    return st.session_state.conversations[0]


def set_active_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id


def start_new_chat() -> None:
    chat_number = len(st.session_state.conversations) + 1
    new_chat = {"id": f"chat-{chat_number}", "title": "New Chat", "messages": []}
    st.session_state.conversations.append(new_chat)
    st.session_state.active_chat_id = new_chat["id"]
    st.session_state.chat_titles = [conversation["title"] for conversation in st.session_state.conversations]


def sync_active_messages() -> list:
    active_chat = get_active_chat()
    st.session_state.messages = active_chat["messages"]
    st.session_state.chat_titles = [conversation["title"] for conversation in st.session_state.conversations]
    return active_chat["messages"]


def build_chat_title(text: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    return (clean[:32] + ("…" if len(clean) > 32 else "")) if clean else "New Chat"


def queue_prompt(prompt: str) -> None:
    st.session_state.pending_query = prompt


def render_sidebar() -> None:
    with st.sidebar:
        if LOGO_DATA_URI:
            st.markdown(
                f"""
                <div class="sidebar-brand">
                    <img class="sidebar-logo" src="{LOGO_DATA_URI}" alt="NepaGen AI logo" />
                    <div>
                        <div class="sidebar-title">NepaGen AI</div>
                        <div class="sidebar-subtitle">Premium Nepali AI chat grounded in your dataset.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="sidebar-brand">
                    <div style="font-size:1.6rem; line-height:1;">🇳🇵</div>
                    <div>
                        <div class="sidebar-title">NepaGen AI</div>
                        <div class="sidebar-subtitle">Premium Nepali AI chat grounded in your dataset.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("New Chat", use_container_width=True):
            start_new_chat()
            st.rerun()

        st.markdown('<div class="sidebar-label">Chat History</div>', unsafe_allow_html=True)
        active_chat_id = st.session_state.active_chat_id
        for conversation in reversed(st.session_state.conversations):
            button_label = f"▶ {conversation['title']}" if conversation["id"] == active_chat_id else conversation["title"]
            if st.button(button_label, key=f"chat-nav-{conversation['id']}", use_container_width=True):
                set_active_chat(conversation["id"])
                st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">About</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-card"><div class="sidebar-subtitle">A calm, premium Nepali-language AI assistant with grounded RAG responses, fast retrieval, and persistent chat history.</div></div>',
            unsafe_allow_html=True,
        )


def render_brand_header() -> None:
    if LOGO_DATA_URI:
        st.markdown(
            f"""
            <div class="brand-header">
                <div class="brand-shell">
                    <img class="brand-logo" src="{LOGO_DATA_URI}" alt="NepaGen AI logo" />
                    <div class="brand-copy">
                        <div class="brand-eyebrow">Premium Nepali AI Assistant</div>
                        <h1 class="brand-title">NepaGen AI</h1>
                        <p class="brand-tagline">Grounded multilingual RAG experience</p>
                        <p class="brand-subcopy">A calm, premium Nepali-language assistant designed for focused conversation and reliable answers.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="brand-header">
                <div class="brand-shell">
                    <div style="font-size:2.6rem; line-height:1;">🇳🇵</div>
                    <div class="brand-copy">
                        <div class="brand-eyebrow">Premium Nepali AI Assistant</div>
                        <h1 class="brand-title">NepaGen AI</h1>
                        <p class="brand-tagline">Grounded multilingual RAG experience</p>
                        <p class="brand-subcopy">A calm, premium Nepali-language assistant designed for focused conversation and reliable answers.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_empty_state() -> None:
    prompt_chips = ["नेपालको इतिहास", "नेपालको संस्कृति", "हिमाल र प्रकृति", "नेपालको अर्थव्यवस्था"]
    st.markdown(
        """
        <div class="hero-wrap">
            <p class="hero-copy" style="margin-top:0;">नेपाल, संस्कृति, इतिहास, geography, and everyday questions in a premium AI chat experience.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chip_rows = st.columns(2)
    for index, prompt in enumerate(prompt_chips):
        with chip_rows[index % 2]:
            if st.button(prompt, key=f"prompt-chip-{index}", use_container_width=True):
                queue_prompt(prompt)
                st.rerun()


def render_user_message(message: dict, message_index: int) -> None:
    content = html_lib.escape(message["content"])
    with st.container():
        st.markdown(
            f'<div class="message-row message-row--user"><div class="message-card message-card--user"><div class="message-content">{content}</div></div></div>',
            unsafe_allow_html=True,
        )


def _assistant_message_height(answer: str) -> int:
    line_count = max(1, answer.count("\n") + 1)
    estimated_from_text = 120 + line_count * 28 + (len(answer) // 42) * 18
    return min(720, max(160, estimated_from_text))


def render_assistant_message(message: dict, message_index: int) -> None:
    answer = message["content"]
    content = html_lib.escape(answer)
    copy_payload = html_lib.escape(json.dumps(answer), quote=True)
    copy_handler = (
        "(async function(button){try{"
        "const text=JSON.parse(button.getAttribute('data-copy-text')||'\"\"');"
        "if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(text);}"
        "else{const ta=document.createElement('textarea');ta.value=text;ta.setAttribute('readonly','');ta.style.position='fixed';ta.style.opacity='0';document.body.appendChild(ta);ta.focus();ta.select();document.execCommand('copy');ta.remove();}"
        "const previous=button.textContent;button.textContent='✓';setTimeout(function(){button.textContent=previous||'🔗';},900);"
        "}catch(error){console.warn('Clipboard copy failed',error);}})(this); return false;"
    )
    copy_handler = html_lib.escape(copy_handler, quote=True)
    iframe_height = _assistant_message_height(answer)
    components.html(
        f"""
        <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            overflow: hidden;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif, "Noto Sans Devanagari";
        }}
        .message-row {{
            display: flex;
            width: 100%;
            justify-content: flex-start;
        }}
        .message-card {{
            position: relative;
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
            width: fit-content;
            max-width: 720px;
            min-width: 120px;
            padding: 18px 22px;
            padding-top: 2.35rem;
            border-radius: 22px;
            white-space: pre-wrap;
            word-break: break-word;
            overflow-wrap: break-word;
            box-sizing: border-box;
            background: linear-gradient(180deg, rgba(18, 24, 40, 0.98), rgba(14, 20, 34, 0.94));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 4px solid rgba(217, 119, 6, 0.82);
            box-shadow: 0 22px 44px rgba(2, 6, 23, 0.34), 0 0 0 1px rgba(255, 255, 255, 0.02) inset;
        }}
        .message-card::before {{
            content: "";
            position: absolute;
            inset: 0;
            border-radius: inherit;
            background: linear-gradient(135deg, rgba(217, 119, 6, 0.08), transparent 42%);
            pointer-events: none;
        }}
        .message-copy-button {{
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 2;
            width: 2.15rem;
            min-width: 2.15rem;
            height: 2.15rem;
            padding: 0;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(15, 23, 42, 0.92);
            color: #e2e8f0;
            box-shadow: 0 10px 16px rgba(2, 6, 23, 0.18);
            line-height: 1;
            font-size: 0.95rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            white-space: nowrap;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease, box-shadow 0.18s ease;
        }}
        .message-copy-button:hover {{
            border-color: rgba(217, 119, 6, 0.35);
            transform: translateY(-1px);
            box-shadow: 0 12px 18px rgba(2, 6, 23, 0.24);
        }}
        .message-copy-button:active {{ transform: translateY(0); }}
        .message-author {{
            color: rgba(217, 119, 6, 0.86);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            font-weight: 700;
        }}
        .message-content {{
            min-width: 0;
            width: 100%;
            color: #eef2ff;
            font-size: 1rem;
            line-height: 1.75;
            -webkit-font-smoothing: antialiased;
            text-rendering: optimizeLegibility;
        }}
        </style>
        <div class="message-row message-row--assistant">
            <div class="message-card message-card--assistant">
                <button class="message-copy-button" type="button" data-copy-text="{copy_payload}" aria-label="Copy assistant message" onclick="{copy_handler}">🔗</button>
                <div class="message-author">NepaGen AI</div>
                <div class="message-content">{content}</div>
            </div>
        </div>
        """,
        height=iframe_height,
        scrolling=False,
    )


def render_composer() -> str | None:
    composer_text = st.chat_input("नेपालीमा प्रश्न सोध्नुस्...", key="composer_text")
    if composer_text and composer_text.strip():
        return composer_text.strip()
    return None


class _EmptyRetriever:
    def get_relevant_documents(self, query):
        return []

    def invoke(self, query):
        return []


class _SafeRetriever:
    def __init__(self, retriever_obj):
        self._retriever = retriever_obj

    def get_relevant_documents(self, query):
        try:
            return self._retriever.get_relevant_documents(query)
        except Exception as first_error:
            print("[startup] retriever.get_relevant_documents failed:", repr(first_error))
            return []

    def invoke(self, query):
        try:
            return self._retriever.invoke(query)
        except Exception as first_error:
            print("[startup] retriever.invoke failed:", repr(first_error))
            return []


def safe_retrieve_documents(retriever_obj, query: str):
    return retriever_obj.get_relevant_documents(query)


@st.cache_resource(show_spinner=False)
def load_models():
    vectorstore_start = time.perf_counter()
    db = load_vectorstore()
    vectorstore_load_time = time.perf_counter() - vectorstore_start

    if getattr(db.index, "ntotal", 0) > 0:
        retriever = _SafeRetriever(db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}))
    else:
        print("[startup] empty vectorstore fallback active; using no-op retriever")
        retriever = _EmptyRetriever()

    llm_start = time.perf_counter()
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=200)
    llm_load_time = time.perf_counter() - llm_start

    total_load_time = vectorstore_load_time + llm_load_time
    print(f"[startup] vectorstore load: {vectorstore_load_time:.2f}s")
    print(f"[startup] embedding model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print(f"[startup] llm init: {llm_load_time:.2f}s")
    print(f"[startup] total load_models: {total_load_time:.2f}s")
    return retriever, llm


try:
    retriever, llm = load_models()
except VectorstoreCompatibilityError as exc:
    st.error(str(exc))
    st.warning("Please rebuild the vectorstore with the current embedding model and redeploy.")
    st.stop()
except Exception as exc:
    st.error("NepaGen AI could not load the FAISS vectorstore. Rebuild it with `python src/ingest.py` and redeploy.")
    st.exception(exc)
    st.stop()

print(f"[startup] total startup time: {time.perf_counter() - STARTUP_START:.2f}s")
print(f"[startup] app revision: {APP_REVISION}")


init_chat_state()
render_sidebar()
messages = sync_active_messages()

render_brand_header()

st.markdown('<div class="chat-layout"><div class="messages-wrapper">', unsafe_allow_html=True)

if not messages:
    render_empty_state()

for message_index, message in enumerate(messages):
    if message["role"] == "assistant":
        render_assistant_message(message, message_index)
    else:
        render_user_message(message, message_index)

st.markdown('</div>', unsafe_allow_html=True)

query = render_composer()
pending_query = st.session_state.pop("pending_query", None)
if not query and pending_query:
    query = pending_query

if query:
    active_chat = get_active_chat()
    if active_chat["title"] == "New Chat":
        active_chat["title"] = build_chat_title(query)
        st.session_state.chat_titles = [conversation["title"] for conversation in st.session_state.conversations]

    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        category, answer = generate_hybrid_reply(query, active_chat)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
