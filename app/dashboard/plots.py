"""Plotly figures with a consistent scientific style."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.preprocessing import EDGE_SAMPLES, FS

from .pipeline import PREDICTION_THRESHOLD
from .theme import FONT_STACK, GRID, INK, MODEL_COLORS, MUTED, PRIMARY, TEXT, TONES

AMPLITUDE_LABEL = "Acceleration (recorded units)"
MAX_PLOT_POINTS = 6000
HEALTHY = TONES["healthy"]["border"]
FAULT = TONES["fault"]["border"]


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def _layout(fig, title, x_title="", y_title="", height=300, hovermode="closest"):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=INK)),
        height=height,
        margin=dict(l=60, r=24, t=52, b=48),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT_STACK, size=12, color=TEXT),
        showlegend=False,
        hovermode=hovermode,
    )
    fig.update_xaxes(title_text=x_title, gridcolor=GRID, linecolor="#9AA7B4", ticks="outside", zeroline=False)
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, linecolor="#9AA7B4", ticks="outside", zeroline=False)
    return fig


def _decimate_minmax(t, x, max_points=MAX_PLOT_POINTS):
    """Keep each bucket's min and max so short impulses stay visible."""
    n = len(x)
    if n <= max_points:
        return t, x
    buckets = max_points // 2
    size = n // buckets
    block = x[: buckets * size].reshape(buckets, size)
    lo, hi = block.argmin(axis=1), block.argmax(axis=1)
    base = np.arange(buckets) * size
    idx = np.empty(2 * buckets, dtype=int)
    idx[0::2] = base + np.minimum(lo, hi)
    idx[1::2] = base + np.maximum(lo, hi)
    return t[idx], x[idx]


def signal_window_figure(values, i0, i1, total_samples, *, title, y_title, color):
    t = np.arange(i0, i1) / FS
    t_plot, v_plot = _decimate_minmax(t, np.asarray(values[i0:i1]))
    fig = go.Figure(
        go.Scatter(x=t_plot, y=v_plot, mode="lines", line=dict(color=color, width=1),
                   hovertemplate="t = %{x:.4f} s<br>%{y:.4g}<extra></extra>")
    )
    edge, end = EDGE_SAMPLES / FS, total_samples / FS
    for x0, x1 in ((0, edge), (end - edge, end)):
        fig.add_vrect(x0=x0, x1=x1, fillcolor="#8795A6", opacity=0.15, line_width=0, layer="below")
    _layout(fig, title, "Time (s)", y_title, height=290)
    fig.update_xaxes(range=[i0 / FS, i1 / FS])
    return fig


def probability_figure(results):
    ok = [r for r in results if r.ok]
    pct = [r.probability * 100 for r in ok]
    fig = go.Figure(
        go.Bar(
            x=pct, y=[r.spec.name for r in ok], orientation="h",
            marker_color=[FAULT if r.prediction == 1 else HEALTHY for r in ok],
            text=[f"{v:.1f}%" for v in pct], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.add_vline(x=PREDICTION_THRESHOLD * 100, line_dash="dash", line_color=MUTED,
                  annotation_text=f"Decision threshold ({PREDICTION_THRESHOLD * 100:.0f}%)",
                  annotation_position="top")
    _layout(fig, "Inner Race fault probability by model", "Inner Race fault probability (%)", "", height=290)
    fig.update_xaxes(range=[0, 105])
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def inference_time_figure(results):
    ok = [r for r in results if r.ok and r.inference_ms is not None]
    fig = go.Figure(
        go.Bar(
            x=[r.spec.name for r in ok], y=[r.inference_ms for r in ok],
            marker_color=[MODEL_COLORS[r.spec.key] for r in ok],
            text=[f"{r.inference_ms:.2f} ms" for r in ok], textposition="outside", cliponaxis=False,
            hovertemplate="%{x}: %{y:.3f} ms<extra></extra>",
        )
    )
    _layout(fig, "Inference time for this recording", "", "Time (ms)", height=290)
    fig.update_xaxes(showgrid=False)
    return fig


def cnn_segments_figure(result):
    probs = np.asarray(result.segment_probabilities) * 100
    x = np.arange(1, len(probs) + 1)
    fig = go.Figure(
        go.Bar(
            x=x, y=probs,
            marker_color=[FAULT if p >= PREDICTION_THRESHOLD * 100 else HEALTHY for p in probs],
            hovertemplate="Segment %{x}: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(y=PREDICTION_THRESHOLD * 100, line_dash="dash", line_color=MUTED)
    fig.add_hline(y=result.probability * 100, line_color=INK, line_width=1.5,
                  annotation_text=f"Recording mean {result.probability * 100:.1f}%",
                  annotation_position="top left")
    _layout(fig, "1D CNN: fault probability per one-second segment",
            "Segment number (after edge trimming)", "Inner Race fault probability (%)", height=300)
    fig.update_yaxes(range=[0, 105])
    fig.update_xaxes(dtick=1 if len(probs) <= 20 else None)
    return fig


def feature_segments_figure(segment_features):
    specs = (
        ("rms", "RMS (recorded units)"),
        ("crest_factor", "Crest factor (dimensionless)"),
        ("envelope_dominant_frequency", "Envelope dominant frequency (Hz)"),
    )
    fig = make_subplots(rows=1, cols=3, subplot_titles=[label for _, label in specs], horizontal_spacing=0.09)
    x = np.arange(1, len(segment_features) + 1)
    for col, (name, _label) in enumerate(specs, start=1):
        values = segment_features[name].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(x=x, y=values, mode="lines+markers", line=dict(color=PRIMARY, width=1.5),
                       marker=dict(size=6), hovertemplate="Segment %{x}: %{y:.4g}<extra></extra>"),
            row=1, col=col,
        )
        fig.add_hline(y=float(np.median(values)), line_dash="dash", line_color=MUTED, row=1, col=col)
        fig.update_xaxes(title_text="Segment number", dtick=1 if len(x) <= 12 else None, row=1, col=col)
    _layout(fig, "Per-segment feature values (dashed line: recording median supplied to LR and RF)", height=320)
    fig.update_annotations(font=dict(size=12, color=TEXT))
    return fig