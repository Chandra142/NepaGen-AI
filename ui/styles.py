"""Responsive visual polish for NepaGen AI.

Base colors and spacing come from .streamlit/config.toml. These rules cover
only the custom hero, prompt cards, chat overflow, and small-screen behavior.
"""
import streamlit as st

APP_CSS = """
:root {
    color-scheme: dark;
    --nepagen-bg: #050812;
    --nepagen-surface: #0D1525;
    --nepagen-surface-hover: #131E35;
    --nepagen-text: #F8FAFC;
    --nepagen-muted: #94A3B8;
    --nepagen-cyan: #22D3EE;
    --nepagen-border: rgba(255, 255, 255, 0.09);
}

.stApp {
    background:
        radial-gradient(circle at 50% -10%, rgba(34, 211, 238, 0.07), transparent 42%),
        var(--nepagen-bg);
}

/* Branded empty state */
.st-key-hero {
    padding: clamp(1.5rem, 6vh, 4.5rem) 0 1.5rem;
    text-align: center;
}
.st-key-hero h1 {
    margin: 0.75rem 0 0.5rem;
    font-size: clamp(2.2rem, 7vw, 3.6rem);
    line-height: 1.08;
    letter-spacing: -0.035em;
    background: linear-gradient(135deg, #FFFFFF, #94A3B8);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.st-key-hero img {
    max-width: 84px;
    border-radius: 20px;
    box-shadow: 0 0 32px rgba(34, 211, 238, 0.18);
}
.st-key-hero p {
    max-width: min(100%, 36rem);
    margin-left: auto;
    margin-right: auto;
    white-space: normal;
    overflow-wrap: anywhere;
}
.st-key-chat-header {
    padding: 0.35rem 0 0.65rem;
    border-bottom: 1px solid var(--nepagen-border);
}

/* Native grid buttons become readable cards on every viewport. */
[class*="st-key-prompt-chip-"] button {
    min-height: 88px;
    padding: 1rem 1.1rem;
    white-space: normal;
    text-align: left;
    align-items: flex-start;
    justify-content: flex-start;
    line-height: 1.45;
    background: var(--nepagen-surface);
    border: 1px solid var(--nepagen-border);
}
@media (hover: hover) {
    [class*="st-key-prompt-chip-"] button:hover {
        background: var(--nepagen-surface-hover);
        border-color: rgba(34, 211, 238, 0.35);
    }
}

/* Sidebar navigation remains usable with long Nepali chat titles. */
.st-key-sidebar-new-chat button,
[class*="st-key-chat-nav-"] button {
    min-height: 44px;
}
[class*="st-key-chat-nav-"] button {
    justify-content: flex-start;
    text-align: left;
}
[class*="st-key-chat-nav-"] button p {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* Prevent long Nepali passages, links, and code from forcing horizontal scroll. */
[data-testid="stChatMessage"] {
    padding: 0.55rem 0;
}
[data-testid="stChatMessage"] .stMarkdown,
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    overflow-wrap: anywhere;
    word-break: normal;
}
.stMarkdown pre {
    max-width: 100%;
    overflow-x: auto;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}
.stMarkdown code {
    overflow-wrap: anywhere;
}
img {
    max-width: 100%;
    height: auto;
}

/* Mobile browsers do not zoom the composer when its font is at least 16px. */
.st-key-composer_text textarea {
    min-height: 54px;
    font-size: 16px !important;
    line-height: 1.45 !important;
    border-radius: 18px !important;
}
.st-key-composer_text {
    padding-bottom: max(0.5rem, env(safe-area-inset-bottom));
}

@media (max-width: 768px) {
    .st-key-hero {
        padding-top: 1.5rem;
    }
    .st-key-hero h1 {
        font-size: clamp(2rem, 10vw, 2.8rem);
    }
    [class*="st-key-prompt-chip-"] button {
        min-height: 72px;
        padding: 0.85rem;
        font-size: 0.92rem;
    }
}

@media (max-width: 480px) {
    .st-key-hero {
        padding: 1rem 0 0.75rem;
    }
    .st-key-hero h1 {
        font-size: clamp(1.9rem, 11vw, 2.5rem);
    }
    .st-key-hero img {
        max-width: 68px;
        border-radius: 16px;
    }
    [class*="st-key-prompt-chip-"] button {
        min-height: 64px;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
"""


def inject_styles() -> None:
    """Inject the small responsive CSS layer into the current page."""
    st.html(f"<style>{APP_CSS}</style>")
