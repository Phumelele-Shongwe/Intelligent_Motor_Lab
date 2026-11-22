"""Small reusable HTML components (cards, banners, pills)."""

import html

import streamlit as st

from .theme import TONES


def _e(value) -> str:
    return html.escape(str(value))


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="imc-header"><div class="imc-title">{_e(title)}</div>'
        f'<p class="imc-sub">{_e(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def section(title: str, description: str = "") -> None:
    st.markdown(f'<div class="imc-section">{_e(title)}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<p class="imc-section-desc">{_e(description)}</p>', unsafe_allow_html=True)


def pill(text: str, tone: str = "neutral") -> str:
    t = TONES[tone]
    return (
        f'<span class="imc-pill" style="color:{t["fg"]};background:{t["bg"]};'
        f'border-color:{t["border"]}">{_e(text)}</span>'
    )


def banner(tone: str, title: str, detail: str = "") -> None:
    t = TONES[tone]
    detail_html = f'<div class="imc-banner-detail">{_e(detail)}</div>' if detail else ""
    st.markdown(
        f'<div class="imc-banner" style="background:{t["bg"]};border-color:{t["border"]}">'
        f'<div class="imc-banner-title" style="color:{t["fg"]}">{_e(title)}</div>{detail_html}</div>',
        unsafe_allow_html=True,
    )


def note(tone: str, text: str) -> None:
    t = TONES[tone]
    st.markdown(
        f'<div class="imc-note" style="background:{t["bg"]};border-color:{t["border"]}">{_e(text)}</div>',
        unsafe_allow_html=True,
    )


def metric_card_html(label: str, value: str, sub: str = "", tone: str = "neutral") -> str:
    colour = f' style="color:{TONES[tone]["fg"]}"' if tone != "neutral" else ""
    sub_html = f'<div class="imc-card-sub">{_e(sub)}</div>' if sub else ""
    return (
        f'<div class="imc-card"><div class="imc-card-label">{_e(label)}</div>'
        f'<div class="imc-card-value"{colour}>{_e(value)}</div>{sub_html}</div>'
    )


def metric_cards(cards: list) -> None:
    """Render ``(label, value, sub, tone)`` tuples in equal-width columns."""
    for col, card in zip(st.columns(len(cards)), cards):
        label, value, *rest = card
        sub = rest[0] if len(rest) > 0 else ""
        tone = rest[1] if len(rest) > 1 else "neutral"
        col.markdown(metric_card_html(label, value, sub, tone), unsafe_allow_html=True)


def key_value_table(rows: list) -> None:
    body = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in rows)
    st.markdown(f'<table class="imc-kv">{body}</table>', unsafe_allow_html=True)


def model_card(result) -> str:
    """HTML for one model's prediction (``result`` is a pipeline.ModelResult)."""
    spec = result.spec
    head = (
        f'<div class="imc-card-label">{_e(spec.input_repr)}</div>'
        f'<div class="imc-model-name">{_e(spec.name)}</div>'
    )
    if not result.ok:
        return (
            f'<div class="imc-card">{head}{pill("Unavailable", "warning")}'
            f'<div class="imc-card-sub" style="margin-top:.6rem">{_e(result.error)}</div></div>'
        )
    tone = "fault" if result.prediction == 1 else "healthy"
    p = result.probability * 100
    time_text = f"{result.inference_ms:.2f} ms" if result.inference_ms is not None else "n/a"
    return (
        f'<div class="imc-card">{head}{pill(result.condition, tone)}'
        f'<div class="imc-card-label" style="margin-top:.8rem">Inner Race fault probability</div>'
        f'<div class="imc-card-value" style="color:{TONES[tone]["fg"]}">{p:.2f}%</div>'
        f'<div class="imc-card-sub">Healthy probability {100 - p:.2f}%</div>'
        f'<div class="imc-card-sub">Inference time {time_text}</div></div>'
    )