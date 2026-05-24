import html as html_lib
import json
import re
import time
STARTUP_START = time.perf_counter()


import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from vectorstore_store import load_vectorstore

load_dotenv()

st.set_page_config(page_title="NepaGen AI", page_icon="🇳🇵", layout="wide")

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
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
    .sidebar-brand { margin-bottom: 0.75rem; }
    .sidebar-title { font-size: 1rem; font-weight: 800; margin: 0.1rem 0 0; letter-spacing: -0.01em; }
    .sidebar-subtitle { color: var(--muted); font-size: 0.86rem; margin: 0.15rem 0 0; line-height: 1.5; }
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


def install_chat_interactions() -> None:
        components.html(
                """
                <script>
                (function () {
                    let rootWindow = window;
                    let rootDocument = document;
                    try {
                        if (window.parent && window.parent.document) {
                            rootWindow = window.parent;
                            rootDocument = window.parent.document;
                        }
                    } catch (error) {
                        console.warn('Parent document unavailable for chat interactions', error);
                    }

                    if (rootWindow.__nepagenChatInteractionsInstalled) {
                        return;
                    }
                    rootWindow.__nepagenChatInteractionsInstalled = true;

                    const copyText = async (text, button) => {
                        try {
                            await rootWindow.navigator.clipboard.writeText(text);
                            const previous = button.textContent;
                            button.textContent = '✓';
                            button.style.borderColor = 'rgba(217, 119, 6, 0.55)';
                            setTimeout(() => {
                                button.textContent = previous;
                                button.style.borderColor = '';
                            }, 900);
                        } catch (error) {
                            console.warn('Clipboard copy failed', error);
                        }
                    };

                    rootDocument.addEventListener('click', function (event) {
                        const button = event.target.closest('.message-copy-button');
                        if (!button) return;
                        const raw = button.getAttribute('data-copy-text') || '""';
                        try {
                            const text = JSON.parse(raw);
                            copyText(text, button);
                        } catch (error) {
                            console.warn('Invalid copy payload', error);
                        }
                    }, true);

                    rootDocument.addEventListener('keydown', function (event) {
                        const target = event.target;
                        if (!target || target.tagName !== 'TEXTAREA') return;
                        if (target.placeholder !== 'नेपालीमा प्रश्न सोध्नुस्...') return;
                        if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return;
                        event.preventDefault();
                        const submit = rootDocument.querySelector('button[kind="formSubmitButton"]');
                        if (submit) {
                            submit.click();
                        }
                    }, true);
                })();
                </script>
                """,
                height=0,
                width=0,
        )


def normalize_query_text(query: str) -> str:
    return query.lower().strip().replace("\\", "").replace("?", "")


def normalize_keyword_text(query: str) -> str:
    return re.sub(r"[^\w\s\u0900-\u097F]", "", query).strip()


def init_chat_state() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = [{"id": "chat-1", "title": "New Chat", "messages": []}]
        st.session_state.active_chat_id = "chat-1"
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
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
        st.markdown(
            """
            <div class="sidebar-brand">
                <div style="font-size:1.4rem; line-height:1;">🇳🇵</div>
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


def render_empty_state() -> None:
    prompt_chips = ["नेपालको इतिहास", "नेपालको संस्कृति", "हिमाल र प्रकृति", "नेपालको अर्थव्यवस्था"]
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-emoji">🇳🇵</div>
            <h1 class="hero-title">NepaGen AI</h1>
            <p class="hero-copy">नेपाल, संस्कृति, इतिहास, geography, and everyday questions in a premium AI chat experience.</p>
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


def render_assistant_message(message: dict, message_index: int) -> None:
    answer = message["content"]
    content = html_lib.escape(answer)
    copy_payload = html_lib.escape(json.dumps(answer), quote=True)
    with st.container():
        st.markdown(
            f'<div class="message-row message-row--assistant"><div class="message-card message-card--assistant">'
            f'<button class="message-copy-button" type="button" data-copy-text="{copy_payload}" aria-label="Copy assistant message">🔗</button>'
            f'<div class="message-author">NepaGen AI</div>'
            f'<div class="message-content">{content}</div></div></div>',
            unsafe_allow_html=True,
        )


def render_composer() -> str | None:
    composer_text = st.chat_input("नेपालीमा प्रश्न सोध्नुस्...", key="composer_text")
    if composer_text and composer_text.strip():
        return composer_text.strip()
    return None


@st.cache_resource(show_spinner=False)
def load_models():
    vectorstore_start = time.perf_counter()
    db = load_vectorstore()
    vectorstore_load_time = time.perf_counter() - vectorstore_start

    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

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
except Exception as exc:
    st.error("NepaGen AI could not load the FAISS vectorstore. Rebuild it with `python src/ingest.py` and redeploy.")
    st.exception(exc)
    st.stop()

print(f"[startup] total startup time: {time.perf_counter() - STARTUP_START:.2f}s")


init_chat_state()
render_sidebar()
install_chat_interactions()
messages = sync_active_messages()

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

    normalized_query = normalize_query_text(query)
    small_talk = {
        "hi": "नमस्ते 👋",
        "hello": "नमस्ते 👋",
        "hey": "नमस्ते 👋",
        "bye": "फेरि भेटौँला 👋",
        "good night": "शुभ रात्रि 🌙",
        "good morning": "शुभ प्रभात ☀️",
        "thanks": "स्वागत छ 😊",
        "thank you": "धन्यवाद 😊",
        "ok": "ठिक छ 👍",
    }
    normalized_lookup = normalize_keyword_text(normalized_query)
    ai_keywords = {
        "what is rag": "RAG stands for Retrieval-Augmented Generation.",
        "rag": "RAG stands for Retrieval-Augmented Generation.",
        "who are you": "म NepaGen AI हुँ ।",
    }

    with st.spinner("Thinking..."):
        if normalized_lookup in small_talk:
            answer = small_talk[normalized_lookup]
        elif normalized_lookup in ai_keywords:
            answer = ai_keywords[normalized_lookup]
        else:
            try:
                docs = retriever.get_relevant_documents(query)
            except Exception:
                docs = retriever.invoke(query)

            def _is_noise_text(t: str) -> bool:
                low = t.lower()
                if "read more" in low or "imagekhabar" in low or "wikipedia" in low:
                    return True
                if "http://" in low or "https://" in low or "www." in low:
                    return True
                if "�" in t:
                    return True
                if len(t.strip()) < 40:
                    return True
                return False

            docs_sorted = sorted(docs, key=lambda d: (-float(d.metadata.get("quality_score", 0)), -len(d.page_content)))
            filtered_docs = [d for d in docs_sorted if not _is_noise_text(d.page_content)]
            query_terms = {term for term in re.findall(r"[\w\u0900-\u097F]+", normalized_query) if len(term) > 1}

            def _rerank(doc):
                content = normalize_query_text(doc.page_content.lower())
                content_terms = {term for term in re.findall(r"[\w\u0900-\u097F]+", content) if len(term) > 1}
                overlap = len(query_terms & content_terms)
                quality = float(doc.metadata.get("quality_score", 0))
                return (-overlap, -quality, -len(doc.page_content))

            filtered_docs = sorted(filtered_docs, key=_rerank)
            top_docs = filtered_docs[:8]
            context = "\n\n".join([d.page_content for d in top_docs])

            prompt = f"""
You are a factual Nepali QA assistant.

Answer ONLY using reliable information from the context.

Ignore:
- opinions
- misleading claims
- incomplete sentences

If the answer is unclear, say:
"मलाई जानकारी भेटिएन।"

Keep answers:
- short
- factual
- direct

Context:
{context}

Question:
{query}

Correct Nepali Answer:
"""

            response = llm.invoke(prompt)
            answer = response.content

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
