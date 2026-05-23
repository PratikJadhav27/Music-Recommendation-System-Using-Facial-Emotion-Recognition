"""
Streamlit custom themes (injected CSS).
Matches the indigo/emerald palette used in emotion_charts.py.
"""

from __future__ import annotations

import streamlit as st

THEME_OPTIONS = ("Studio Dark", "Default (Streamlit)")

STUDIO_DARK_CSS = """
/* App shell */
.stApp {
    background: linear-gradient(165deg, #0e1117 0%, #111827 42%, #0f172a 100%);
}
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1100px;
}
[data-testid="stHeader"] {
    background: transparent;
}
hr {
    border-color: #374151 !important;
    opacity: 0.6;
}

/* Typography */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    background: linear-gradient(92deg, #a5b4fc 0%, #818cf8 35%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
h2, h3, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
    color: #f3f4f6 !important;
}
p, label, .stCaption, [data-testid="stMarkdownContainer"] p {
    color: #d1d5db;
}
[data-testid="stMarkdownContainer"] strong {
    color: #f9fafb;
}
a {
    color: #a5b4fc !important;
}
a:hover {
    color: #34d399 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0e1117 100%);
    border-right: 1px solid #374151;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #e5e7eb !important;
}

/* Inputs */
.stSelectbox label, .stSlider label, .stCheckbox label, .stRadio label,
.stFileUploader label, .stNumberInput label {
    color: #e5e7eb !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #1f2937 !important;
    border-color: #4b5563 !important;
    color: #f3f4f6 !important;
}
.stSlider [data-baseweb="slider"] div {
    color: #818cf8;
}

/* Primary buttons */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #f9fafb !important;
    border: 1px solid #6366f1 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.stButton > button:hover {
    border-color: #818cf8 !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
}
.stButton > button:active {
    transform: scale(0.98);
}
.stButton > button:disabled {
    opacity: 0.45 !important;
    box-shadow: none !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
    border: 1px solid #374151;
}

/* Expander */
details {
    background-color: #1f2937;
    border-radius: 8px;
    border: 1px solid #374151;
}
"""


def inject_theme(theme_name: str) -> None:
    """Apply custom CSS when theme is not the Streamlit default."""
    if theme_name != "Studio Dark":
        return
    st.markdown(f"<style>{STUDIO_DARK_CSS}</style>", unsafe_allow_html=True)


def render_hero_subtitle() -> None:
    """Short tagline under the main title."""
    st.markdown(
        '<p class="app-hero-subtitle" style="margin-top:-0.5rem;color:#9ca3af;font-size:1.05rem;">'
        "Detect your mood from a photo — get songs that match how you feel."
        "</p>",
        unsafe_allow_html=True,
    )
