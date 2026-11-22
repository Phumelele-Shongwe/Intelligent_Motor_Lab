"""Visual identity: palette, typography and global CSS."""

import streamlit as st

INK = "#1B2733"
TEXT = "#2D3B4A"
MUTED = "#5F7080"
BORDER = "#D5DBE1"
GRID = "#E6EAEE"
SURFACE = "#F4F6F8"
PRIMARY = "#27496D"

FONT_STACK = "Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

# Status colours carry meaning: green = healthy, red = fault,
# amber = warning / uncertainty, grey-blue = neutral information.
TONES = {
    "healthy": {"fg": "#1E6B41", "bg": "#E8F3EC", "border": "#2E7D4F"},
    "fault": {"fg": "#9C2626", "bg": "#FBEAEA", "border": "#B83232"},
    "warning": {"fg": "#7A4E08", "bg": "#FBF3E2", "border": "#B7791F"},
    "neutral": {"fg": TEXT, "bg": "#EEF2F6", "border": "#8795A6"},
}

# Signal-trace colours (deliberately not green/red/amber).
SIGNAL_COLORS = {"raw": "#4A5A6A", "filtered": "#27496D", "envelope": "#2A7F8E"}
MODEL_COLORS = {
    "logistic_regression": "#27496D",
    "random_forest": "#6B7F94",
    "cnn": "#2A7F8E",
}

_CSS = f"""
<style>
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }}
#MainMenu, footer {{ visibility: hidden; }}
html, body, [class*="css"] {{ font-family: {FONT_STACK}; }}
section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}

.imc-header {{ border-bottom: 1px solid {BORDER}; padding-bottom: .9rem; margin-bottom: 1.1rem; }}
.imc-title {{ font-size: 1.6rem; font-weight: 650; color: {INK}; margin: 0; letter-spacing: -0.01em; }}
.imc-sub {{ color: {MUTED}; font-size: .95rem; margin: .25rem 0 0; }}

.imc-section {{ font-size: 1.15rem; font-weight: 650; color: {INK}; margin: 1.4rem 0 .15rem; }}
.imc-section-desc {{ color: {MUTED}; font-size: .9rem; margin: 0 0 .7rem; }}

.imc-card {{ border: 1px solid {BORDER}; border-radius: 6px; background: #fff; padding: .85rem 1rem; min-height: 6.2rem; }}
.imc-card-label {{ font-size: .82rem; color: {MUTED}; font-weight: 600; }}
.imc-card-value {{ font-size: 1.55rem; font-weight: 650; color: {INK}; line-height: 1.25; margin-top: .2rem; }}
.imc-card-sub {{ font-size: .82rem; color: {MUTED}; margin-top: .2rem; }}
.imc-model-name {{ font-size: 1.05rem; font-weight: 650; color: {INK}; margin: .1rem 0 .5rem; }}

.imc-banner {{ border-left: 5px solid; border-radius: 4px; padding: .85rem 1.1rem; margin: .2rem 0 1rem; }}
.imc-banner-title {{ font-size: 1.2rem; font-weight: 650; }}
.imc-banner-detail {{ font-size: .92rem; margin-top: .2rem; color: {TEXT}; }}

.imc-note {{ border-left: 3px solid; border-radius: 3px; padding: .55rem .85rem; margin: .5rem 0 .9rem; font-size: .88rem; color: {TEXT}; }}
.imc-pill {{ display: inline-block; padding: .1rem .65rem; border-radius: 999px; font-size: .8rem; font-weight: 600; border: 1px solid; }}

.imc-kv {{ width: 100%; font-size: .85rem; border-collapse: collapse; }}
.imc-kv td {{ padding: .18rem 0; vertical-align: top; }}
.imc-kv td:first-child {{ color: {MUTED}; padding-right: .8rem; white-space: nowrap; }}
.imc-kv td:last-child {{ color: {INK}; font-weight: 600; word-break: break-all; }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)