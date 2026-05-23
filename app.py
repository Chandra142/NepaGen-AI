import streamlit as st
from dotenv import load_dotenv
import html as html_lib
import json

import streamlit.components.v1 as components

from langchain_groq import ChatGroq

from vectorstore_store import load_vectorstore

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NepaGen AI",
    page_icon="🇳🇵",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

:root {
    --page-bg: #0f172a;
    --bubble-bg: linear-gradient(160deg, rgba(30, 41, 59, 0.98) 0%, rgba(15, 23, 42, 0.98) 55%, rgba(17, 24, 39, 0.98) 100%);
    --bubble-accent: linear-gradient(135deg, rgba(37, 99, 235, 0.22) 0%, rgba(124, 58, 237, 0.18) 45%, rgba(14, 165, 233, 0.12) 100%);
    --bubble-border: rgba(96, 165, 250, 0.28);
    --bubble-shadow: 0 0.9rem 2.5rem rgba(2, 6, 23, 0.42), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    --text-primary: #e2e8f0;
    --text-muted: #cbd5e1;
}

@keyframes bg-drift {
    0% {
        transform: translate3d(0, 0, 0) scale(1);
        opacity: 0.55;
    }
    50% {
        transform: translate3d(1.5rem, -1rem, 0) scale(1.05);
        opacity: 0.75;
    }
    100% {
        transform: translate3d(0, 0, 0) scale(1);
        opacity: 0.55;
    }
}

@keyframes bg-slow-pan {
    0% {
        background-position: 0% 0%, 100% 0%, 50% 50%;
    }
    50% {
        background-position: 8% 10%, 92% 8%, 50% 50%;
    }
    100% {
        background-position: 0% 0%, 100% 0%, 50% 50%;
    }
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

@keyframes typing-bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.65; }
    40% { transform: translateY(-0.3rem); opacity: 1; }
}

@keyframes message-fade {
    from { opacity: 0; transform: translateY(0.35rem) scale(0.995); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

html, body {
    background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(124, 58, 237, 0.16), transparent 28%),
        linear-gradient(180deg, #020617 0%, #0f172a 45%, #111827 100%);
    background-attachment: fixed;
}

.main {
    background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(124, 58, 237, 0.12), transparent 24%),
        linear-gradient(180deg, #020617 0%, #0f172a 48%, #111827 100%);
    color: var(--text-primary);
    position: relative;
    isolation: isolate;
}

.main::before,
.main::after {
    content: "";
    position: fixed;
    inset: auto;
    width: 22rem;
    height: 22rem;
    border-radius: 50%;
    pointer-events: none;
    filter: blur(28px);
    opacity: 0.6;
    z-index: -1;
    animation: bg-drift 18s ease-in-out infinite;
}

.main::before {
    top: -5rem;
    left: -6rem;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.28), transparent 70%);
}

.main::after {
    bottom: -6rem;
    right: -5rem;
    background: radial-gradient(circle, rgba(124, 58, 237, 0.24), transparent 70%);
    animation-duration: 22s;
    animation-direction: reverse;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(124, 58, 237, 0.12), transparent 24%),
        linear-gradient(180deg, #020617 0%, #0f172a 48%, #111827 100%);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(148, 163, 184, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148, 163, 184, 0.04) 1px, transparent 1px);
    background-size: 3rem 3rem;
    mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.75), transparent 95%);
    opacity: 0.45;
    animation: bg-slow-pan 26s ease-in-out infinite;
}

.block-container {
    max-width: 70vw;
    padding-top: clamp(1rem, 2.5vw, 2rem);
    padding-bottom: clamp(1rem, 2.5vw, 2rem);
    padding-left: clamp(0.75rem, 2vw, 2rem);
    padding-right: clamp(0.75rem, 2vw, 2rem);
    margin-left: auto;
    margin-right: auto;
}

/* Sidebar styles removed for a cleaner single-column chat layout. */

.stChatInput input {
    background-color: #1e293b !important;
    color: var(--text-primary) !important;
    border-radius: 999px !important;
    padding: 0.9rem 1.1rem !important;
    font-size: 1rem !important;
}

.stChatInput input::placeholder {
    color: #94a3b8 !important;
}

.title {
    text-align: center;
    font-size: clamp(2rem, 5vw, 3.25rem);
    font-weight: bold;
    background: linear-gradient(to right, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.65rem;
}

.subtitle {
    text-align: center;
    color: var(--text-muted);
    margin-bottom: clamp(1.5rem, 4vw, 2.5rem);
    font-size: clamp(0.95rem, 2vw, 1.125rem);
    line-height: 1.45;
    padding-inline: 1rem;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    margin-bottom: 0.9rem;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(96, 165, 250, 0.16);
    color: #bfdbfe;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

        .assistant-copy {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    width: 100%;
    max-width: 100%;
    padding: 1rem 1rem 1.1rem 1rem;
    border-radius: 1.125rem;
    border: 1px solid rgba(96, 165, 250, 0.22);
    background: linear-gradient(160deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.94));
    box-shadow: 0 0.85rem 2.3rem rgba(2, 6, 23, 0.28);
    overflow: hidden;
            text-decoration: none;
            appearance: none;
            -webkit-appearance: none;

            cursor: pointer;
            outline: none;
.assistant-loading::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.06), transparent);
    transform: translateX(-100%);
    animation: shimmer 1.8s ease-in-out infinite;
}

.loading-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.loading-avatar {
    width: 2rem;
    height: 2rem;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.95), rgba(124, 58, 237, 0.9));
    color: white;
    font-size: 1rem;
    box-shadow: 0 0.65rem 1.4rem rgba(37, 99, 235, 0.22);
}

.loading-label {
    color: #e2e8f0;
    font-weight: 600;
    font-size: 0.95rem;
}

.typing-dots {
    display: inline-flex;
    gap: 0.35rem;
    align-items: center;
}

.typing-dots span {
    width: 0.48rem;
    height: 0.48rem;
    border-radius: 999px;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    animation: typing-bounce 1.1s infinite ease-in-out;
}

.typing-dots span:nth-child(2) {
    animation-delay: 0.15s;
}

.typing-dots span:nth-child(3) {
    animation-delay: 0.3s;
}

.skeleton-line {
    width: 100%;
    height: 0.8rem;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(148, 163, 184, 0.12), rgba(148, 163, 184, 0.22), rgba(148, 163, 184, 0.12));
    background-size: 200% 100%;
    animation: shimmer 1.9s ease-in-out infinite;
}

.skeleton-line.short {
    width: 72%;
}

.skeleton-line.medium {
    width: 88%;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem;
    border-radius: 1.15rem;
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.96));
    border: 1px solid rgba(96, 165, 250, 0.14);
    margin-bottom: 1rem;
}

.sidebar-brand-mark {
    width: 2.4rem;
    height: 2.4rem;
    border-radius: 0.85rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.95), rgba(124, 58, 237, 0.92));
    color: white;
    font-weight: 800;
}

.sidebar-card {
    padding: 0.95rem 1rem;
    border-radius: 1rem;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.12);
    color: #cbd5e1;
    margin-bottom: 0.8rem;
}

.sidebar-card h4 {
    margin: 0 0 0.35rem 0;
    color: #f8fafc;
    font-size: 0.95rem;
}

.sidebar-card p {
    margin: 0;
    font-size: 0.85rem;
    line-height: 1.5;
    color: #cbd5e1;
}

.page-footer {
    margin-top: clamp(1rem, 2vw, 1.5rem);
    padding: 0.9rem 1rem;
    text-align: center;
    color: #94a3b8;
    font-size: 0.86rem;
}

@media (max-width: 48rem) {
    .block-container {
        max-width: 100%;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }

    .title {
        font-size: clamp(1.7rem, 9vw, 2.6rem);
    }

    .subtitle {
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .assistant-loading {
        padding: 0.9rem;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown(
    '''
    <div class="hero-shell" style="text-align:center;">
        <div class="title">🇳🇵 NepaGen AI</div>
    </div>
    ''' ,
    unsafe_allow_html=True
)


def normalize_query_text(query: str) -> str:

    return (
        query.lower()
        .strip()
        .replace("\\", "")
    )


def render_loading_indicator() -> str:

    return """
    <div class="assistant-loading">
        <div class="loading-row">
            <div class="loading-avatar">AI</div>
            <div>
                <div class="loading-label">Thinking...</div>
                <div class="typing-dots" aria-label="Generating response">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
        <div class="skeleton-line short"></div>
        <div class="skeleton-line medium"></div>
        <div class="skeleton-line short"></div>
    </div>
    """


def render_assistant_message(message: dict) -> None:

    with st.chat_message("assistant"):

        answer = message["content"]
        escaped_answer = html_lib.escape(answer)
        answer_json = json.dumps(answer)

        bubble_html = """
        <style>
        .assistant-bubble {{
            position: relative;
            display: flex;
            flex-direction: column;
            width: 100%;
            max-width: 100%;
            min-height: fit-content;
            height: auto;
            padding: 1rem 2.9rem 1rem 1rem;
            border: 1px solid rgba(96, 165, 250, 0.28);
            border-radius: 1.125rem;
            background: linear-gradient(160deg, rgba(30, 41, 59, 0.98) 0%, rgba(15, 23, 42, 0.98) 55%, rgba(17, 24, 39, 0.98) 100%);
            box-shadow: 0 0.9rem 2.5rem rgba(2, 6, 23, 0.42), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            animation: message-fade 0.35s ease;
            overflow: visible;
        }}

        .assistant-bubble:hover {{
            transform: translateY(-1px);
            border-color: rgba(96, 165, 250, 0.56);
            box-shadow: 0 1rem 3rem rgba(37, 99, 235, 0.18), 0 1.2rem 3rem rgba(2, 6, 23, 0.46), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }}

        .assistant-text {{
            width: 100%;
            max-width: 100%;
            height: auto;
            min-height: fit-content;
            color: #f1f5f9;
            font-size: clamp(0.95rem, 1.8vw, 1.05rem);
            line-height: 1.75;
            white-space: pre-wrap;
            word-break: break-word;
            overflow-wrap: break-word;
            padding-right: 2.25rem;
        }}

        @media (max-width: 48rem) {{
            .assistant-bubble {{
                padding: 0.9rem 2.7rem 0.9rem 0.9rem;
                border-radius: 1rem;
            }}
        }}
        </style>

        <div class="assistant-bubble">
            <div class="assistant-text">{escaped_answer}</div>
        </div>
        """.format(answer_json=answer_json, escaped_answer=escaped_answer)

        st.markdown(bubble_html, unsafe_allow_html=True)

        copy_html = """
        <style>
        html, body {
            margin: 0;
            padding: 0;
            background: transparent;
            overflow: hidden;
        }

        .copy-wrap {
            width: 100%;
            display: flex;
            justify-content: flex-end;
            padding-top: 0.35rem;
            box-sizing: border-box;
        }

        .copy-btn {
            width: 1.9rem;
            height: 1.9rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.24);
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.78), rgba(30, 41, 59, 0.82));
            color: #dbeafe;
            cursor: pointer;
            transition: transform 0.18s ease, background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
            box-shadow: 0 0.45rem 1rem rgba(2, 6, 23, 0.32);
            font-size: 0.95rem;
            line-height: 1;
        }

        .copy-btn:hover {
            transform: scale(1.08);
            background: linear-gradient(180deg, rgba(37, 99, 235, 0.38), rgba(124, 58, 237, 0.28));
            border-color: rgba(96, 165, 250, 0.92);
            color: #ffffff;
            box-shadow: 0 0.75rem 1.6rem rgba(37, 99, 235, 0.3), 0 0 0 1px rgba(96, 165, 250, 0.2);
        }

        .copy-btn:active {
            transform: scale(0.96);
        }
        </style>

        <div class="copy-wrap">
            <button id="copy-btn" class="copy-btn" type="button" title="Copy answer" aria-label="Copy answer">🔗</button>
        </div>

        <script>
        (function() {
            const button = document.getElementById("copy-btn");
            const text = __ANSWER_TEXT__;

            async function copyText() {
                try {
                    if (navigator.clipboard && window.isSecureContext) {
                        await navigator.clipboard.writeText(text);
                    } else {
                        const textarea = document.createElement("textarea");
                        textarea.value = text;
                        textarea.style.position = "fixed";
                        textarea.style.opacity = "0";
                        document.body.appendChild(textarea);
                        textarea.focus();
                        textarea.select();
                        document.execCommand("copy");
                        document.body.removeChild(textarea);
                    }

                    button.textContent = "✓";
                    setTimeout(() => {
                        button.textContent = "🔗";
                    }, 1200);
                } catch (error) {
                    console.error(error);
                }
            }

            button.addEventListener("click", copyText);
        })();
        </script>
        """.replace("__ANSWER_TEXT__", answer_json)

        components.html(copy_html, height=42, scrolling=False)
# =========================
# SIDEBAR
# =========================
# Sidebar removed.

# =========================
# CACHE MODELS
# =========================
@st.cache_resource
def load_models():
    # Vector DB
    db = load_vectorstore()

    # Retriever
    retriever = db.as_retriever(
        search_kwargs={"k": 5}
    )

    # LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=200
    )

    return retriever, llm

try:
    retriever, llm = load_models()
except Exception as exc:
    st.error("NepaGen AI could not load the FAISS vectorstore. Rebuild it with `python src/ingest.py` and redeploy.")
    st.exception(exc)
    st.stop()

# =========================
# CHAT HISTORY
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# DISPLAY OLD MESSAGES
# =========================
for message in st.session_state.messages:

    if message["role"] == "assistant":

        render_assistant_message(message)

    else:

        with st.chat_message("user"):

            st.markdown(message["content"])

# =========================
# CHAT INPUT
# =========================
query = st.chat_input("Ask your question in Nepali...")

# =========================
# MAIN CHAT FLOW
# =========================
if query:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    normalized_query = normalize_query_text(query)
    # =========================
    # SMALL TALK
    # =========================
    small_talk = {
        "hi": "नमस्ते 👋",
        "hello": "नमस्ते 👋",
        "hey": "नमस्ते 👋",
        "bye": "फेरि भेटौँला 👋",
        "good night": "शुभ रात्रि 🌙",
        "good morning": "शुभ प्रभात ☀️",
        "thanks": "स्वागत छ 😊",
        "thank you": "धन्यवाद 😊",
        "ok": "ठिक छ 👍"
    }

    # =========================
    # AI KEYWORDS
    # =========================
    ai_keywords = {
        "what is rag": "RAG stands for Retrieval-Augmented Generation.",
        "rag": "RAG stands for Retrieval-Augmented Generation.",
        "what is ai": "AI means Artificial Intelligence.",
        "who are you": "म NepaGen AI हुँ ।"
    }

    loading_slot = st.empty()

    with loading_slot.container():

        with st.chat_message("assistant"):

            st.markdown(render_loading_indicator(), unsafe_allow_html=True)

    with st.spinner("Thinking..."):

        # =========================
        # SMALL TALK RESPONSE
        # =========================
        if normalized_query in small_talk:

            answer = small_talk[normalized_query]
            context = ""

        # =========================
        # AI KEYWORD RESPONSE
        # =========================
        elif normalized_query in ai_keywords:

            answer = ai_keywords[normalized_query]
            context = ""

        # =========================
        # RAG PIPELINE
        # =========================
        else:

            # Retrieve documents
            docs = retriever.invoke(query)

            # Filter noisy chunks
            filtered_docs = []

            for doc in docs:

                text = doc.page_content

                if (
                    "चीनिया दावा" in text
                    or "Wikipedia" in text
                    or "Read more" in text
                ):
                    continue

                filtered_docs.append(text)

            # Build context
            context = "\n\n".join(filtered_docs)

            # Prompt
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

            # Generate answer
            response = llm.invoke(prompt)

            answer = response.content

        # =========================
        # STORE ASSISTANT MESSAGE
        # =========================
        loading_slot.empty()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # Refresh UI
        st.rerun()