"""CSS styles for NepaGen AI.

All application CSS lives here so app.py stays clean.
Call inject_styles() once at the top of app.py.
"""
import streamlit as st

APP_CSS = """
:root {
    color-scheme: dark;
    --bg-base: #050812;
    --bg-surface: #0D1525;
    --bg-surface-hover: #131E35;
    --text-primary: #F8FAFC;
    --text-muted: #94A3B8;
    --cyan: #22D3EE;
    --blue: #38BDF8;
    --gold: #F59E0B;
    --deep-navy: #0B1220;
    --border-light: rgba(255, 255, 255, 0.08);
    --border-cyan: rgba(34, 211, 238, 0.3);
    
    /* Animation timings */
    --t-fast: 150ms;
    --t-normal: 250ms;
    --t-slow: 400ms;
    --t-hero: 800ms;
}

/* 1. Global & Dynamic Background */
html, body {
    background: var(--bg-base);
    color: var(--text-primary);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    margin: 0;
    padding: 0;
    min-height: 100vh;
    overflow-x: hidden;
}

.stApp, .main {
    background: transparent !important;
}

/* Animated Ambient Layers */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -2;
    background: radial-gradient(circle at 50% 50%, rgba(34, 211, 238, 0.03) 0%, transparent 60%),
                radial-gradient(circle at 80% 20%, rgba(56, 189, 248, 0.02) 0%, transparent 40%),
                radial-gradient(circle at 20% 80%, rgba(245, 158, 11, 0.015) 0%, transparent 50%);
    background-size: 200% 200%;
    animation: ambientShift 25s ease-in-out infinite alternate;
    pointer-events: none;
}
.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    background-image: linear-gradient(rgba(255,255,255,0.01) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,0.01) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    opacity: 0.5;
}

@keyframes ambientShift {
    0% { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}

/* 2. Block Container & Layout */
.block-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem 1rem 8rem !important;
}

/* 3. Top Navigation Bar */
.top-nav {
    position: sticky;
    top: 0;
    z-index: 50;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1.5rem;
    background: rgba(5, 8, 18, 0.6);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border-light);
    margin: -2rem -1rem 2rem -1rem; /* Negate block-container padding */
    animation: slideDownFade var(--t-normal) ease-out;
}
.top-nav-left {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.top-nav-logo {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    animation: pulseGlow 4s infinite alternate;
}
.top-nav-title {
    font-weight: 600;
    font-size: 1rem;
    color: var(--text-primary);
}
.top-nav-center {
    font-size: 0.9rem;
    color: var(--text-muted);
}
.top-nav-right {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    color: var(--cyan);
}
.rag-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 8px var(--cyan);
}

/* 4. Hero Landing Experience */
.hero-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    margin-top: 10vh;
    margin-bottom: 4rem;
    animation: slideUpFade var(--t-hero) cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
.hero-logo {
    width: 80px;
    height: 80px;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 40px rgba(34, 211, 238, 0.15);
    animation: pulseGlow 4s infinite alternate;
}
@keyframes pulseGlow {
    0% { box-shadow: 0 0 20px rgba(34, 211, 238, 0.1); }
    100% { box-shadow: 0 0 50px rgba(34, 211, 238, 0.25); }
}
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FFFFFF, var(--text-muted));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 1.25rem;
    color: var(--cyan);
    margin: 0 0 1rem 0;
    font-weight: 500;
}
.hero-tagline {
    font-size: 1.1rem;
    color: var(--text-muted);
    margin: 0;
}

/* Example Prompt Cards */
.prompt-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin-top: 3rem;
    width: 100%;
    max-width: 800px;
}
.prompt-card-wrapper button {
    height: 100%;
    width: 100%;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 16px !important;
    padding: 1.25rem !important;
    color: var(--text-primary) !important;
    text-align: left !important;
    transition: all var(--t-fast) ease-out !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
}
.prompt-card-wrapper button:hover {
    border-color: var(--border-cyan) !important;
    background: var(--bg-surface-hover) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(34, 211, 238, 0.1) !important;
}
.prompt-card-wrapper button p {
    margin: 0 !important;
    font-size: 1rem !important;
    line-height: 1.5 !important;
}

/* 5. Chat Layout & Messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 1rem 0 !important;
    animation: slideUpFade var(--t-normal) ease-out forwards;
    display: flex;
    gap: 1rem;
}
[data-testid="stChatMessageAvatarUser"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-light);
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stChatMessageAvatarAssistant"] img {
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(34, 211, 238, 0.2);
}

/* User Message Glass Bubble */
[data-testid="stChatMessage"][data-baseweb="card"] { /* Streamlit defaults */ }
.stChatMessage-user {
    justify-content: flex-end;
}
.stChatMessage-user .stMarkdown {
    background: rgba(34, 211, 238, 0.05);
    border: 1px solid rgba(34, 211, 238, 0.1);
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.25rem;
    display: inline-block;
    max-width: 85%;
}

/* Assistant Message Styling */
.stChatMessage-assistant .stMarkdown {
    max-width: 95%;
    color: var(--text-primary);
    line-height: 1.6;
}

/* Markdown typography inside messages */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: var(--text-primary);
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}
.stMarkdown code {
    background: var(--deep-navy) !important;
    color: var(--cyan) !important;
    border-radius: 4px;
    padding: 0.2em 0.4em;
}
.stMarkdown pre {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

/* 6. Composer (Chat Input) */
div[data-testid="stChatInput"] {
    background: transparent !important;
    border: none !important;
    padding: 1rem 0 2rem !important;
}
div[data-testid="stChatInput"] textarea {
    background: rgba(13, 21, 37, 0.8) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 24px !important;
    color: var(--text-primary) !important;
    padding: 1rem 1.5rem !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5) !important;
    backdrop-filter: blur(20px);
    transition: all var(--t-fast) ease-out !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 10px 40px rgba(34, 211, 238, 0.15) !important;
}
div[data-testid="stChatInput"] button {
    background: var(--cyan) !important;
    color: var(--bg-base) !important;
    border-radius: 50% !important;
    transition: all var(--t-fast) ease-out !important;
}
div[data-testid="stChatInput"] button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 0 15px rgba(34, 211, 238, 0.4) !important;
}

/* 7. Sidebar */
section[data-testid="stSidebar"] {
    background: var(--deep-navy) !important;
    border-right: 1px solid var(--border-light);
    width: 280px !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}
/* New Chat Button Container */
.new-chat-container button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    padding: 0.75rem !important;
    color: var(--text-primary) !important;
    transition: all var(--t-fast) ease-out !important;
    font-weight: 600 !important;
    justify-content: center !important;
}
.new-chat-container button:hover {
    background: rgba(34, 211, 238, 0.1) !important;
    border-color: var(--cyan) !important;
    box-shadow: 0 0 15px rgba(34, 211, 238, 0.15) !important;
}

/* Chat History Items (Secondary = Inactive) */
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:not(.new-chat-container button) {
    background: transparent !important;
    border: none !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.6rem 0.75rem !important;
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    transition: all var(--t-fast) ease-out !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
}
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:not(.new-chat-container button):hover {
    background: rgba(255,255,255,0.05) !important;
    color: var(--text-primary) !important;
    transform: translateX(4px) !important;
}

/* Chat History Items (Primary = Active) */
[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
    background: rgba(34, 211, 238, 0.1) !important;
    color: var(--cyan) !important;
    border: none !important;
    border-left: 3px solid var(--cyan) !important;
    border-radius: 4px 8px 8px 4px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.6rem 0.75rem !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    transition: all var(--t-fast) ease-out !important;
}
[data-testid="stSidebar"] button[data-testid="baseButton-primary"]:hover {
    box-shadow: inset 0 0 15px rgba(34, 211, 238, 0.05) !important;
}

/* 8. Animations */
.thinking-dots::after {
    content: '.';
    animation: dots 1.5s steps(4, end) infinite;
}
@keyframes dots {
    0%, 20% { content: ''; }
    40% { content: '.'; }
    60% { content: '..'; }
    80%, 100% { content: '...'; }
}
@keyframes slideUpFade {
    0% { opacity: 0; transform: translateY(12px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes slideDownFade {
    0% { opacity: 0; transform: translateY(-12px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Reduced Motion */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
"""
def inject_styles() -> None:
    """Inject the application CSS into the current Streamlit page."""
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
