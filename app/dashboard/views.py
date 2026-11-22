"""Tab content for the dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.preprocessing import (
    EDGE_SAMPLES,
    ENVELOPE_MAX_FREQ,
    FS,
    HIGH_CUT,
    LOW_CUT,
    WINDOW_SAMPLES,
)

from . import plots
from .components import banner, metric_cards, model_card, note, pill, section
from .pipeline import CLASS_FAULT, CLASS_HEALTHY, MODEL_SPECS, summarise
from .theme import SIGNAL_COLORS

EMPTY_NO_RECORDING = "Select a demonstration recording or upload a .mat file in the sidebar to begin."
EMPTY_NO_ANALYSIS = "Press Run bearing diagnosis in the sidebar to analyse the loaded recording."


# ------------------------------------------------------------- overview

def render_overview(reference_figure: Path) -> None:
    section("Bearing condition from a single vibration recording")
    st.write(
        "This tool assesses the condition of a motor bearing from one drive-end vibration "
        "recording. Three independently trained classifiers analyse the same signal, and their "
        "assessments are shown side by side so that agreement or disagreement is visible."
    )

    section("Diagnostic classes")
    left, right = st.columns(2)
    left.markdown(pill(CLASS_HEALTHY, "healthy") + " &nbsp;No bearing fault detected in the analysed signal.",
                  unsafe_allow_html=True)
    right.markdown(pill(CLASS_FAULT, "fault") + " &nbsp;Fault on the inner raceway of the bearing.",
                   unsafe_allow_html=True)
    note("warning",
         "Scope: the models separate Healthy from Inner Race Fault only. A recording with another fault "
         "type or different operating conditions is still assigned to one of these two classes, so a "
         "Healthy result does not prove that no fault is present.")

    section("Implemented models")
    cols = st.columns(3)
    descriptions = {
        "logistic_regression": "Linear classifier on three engineered features. Fast and compact.",
        "random_forest": "Tree ensemble on the same three features; captures nonlinear relationships.",
        "cnn": "One-dimensional convolutional network that learns directly from raw vibration segments.",
    }
    for col, spec in zip(cols, MODEL_SPECS):
        col.markdown(
            f'<div class="imc-card"><div class="imc-card-label">{spec.input_repr}</div>'
            f'<div class="imc-model-name">{spec.name}</div>'
            f'<div class="imc-card-sub">{descriptions[spec.key]}</div></div>',
            unsafe_allow_html=True,
        )

    section("How a recording is analysed")
    min_seconds = (2 * EDGE_SAMPLES + WINDOW_SAMPLES) / FS
    st.markdown(
        f"1. **Validation.** One-dimensional, finite signal of at least {min_seconds:.1f} s, sampled at {FS / 1000:.0f} kHz.\n"
        f"2. **Segmentation.** {EDGE_SAMPLES / FS:.2f} s is removed from each end and the remainder is split into non-overlapping {WINDOW_SAMPLES / FS:.0f} s segments.\n"
        f"3. **Feature path (Logistic Regression, Random Forest).** {LOW_CUT / 1000:.0f}–{HIGH_CUT / 1000:.0f} kHz band-pass filter and Hilbert envelope; RMS, crest factor and envelope dominant frequency are computed per segment, and their median is classified.\n"
        "4. **Deep-learning path (1D CNN).** Raw segments are normalised with the training-set mean and standard deviation; segment probabilities are averaged into one recording-level probability."
    )

    with st.expander("Offline evaluation of the trained models (reference only)"):
        st.write(
            "These results describe how the models performed on held-out data during development. "
            "They are not about the recording currently loaded, and a prediction probability is not an "
            "accuracy. Perfect scores on a small test set do not establish real-world accuracy; see "
            "results/final_experimental_comparison.md for the size and limitations of the test set."
        )
        if reference_figure.exists():
            st.image(str(reference_figure), width="stretch")
        else:
            st.caption(f"Figure not found: {reference_figure}")


# --------------------------------------------------------------- signal

def render_signal(recording, filtered, envelope) -> None:
    if recording is None:
        st.info(EMPTY_NO_RECORDING)
        return

    signal = recording.signal
    n = len(signal)
    duration = n / FS
    section("Signal analysis", f"Recording: {recording.name} (variable: {recording.variable})")
    metric_cards([
        ("Samples", f"{n:,}"),
        ("Duration", f"{duration:.2f} s"),
        ("Sampling frequency", f"{FS / 1000:.0f} kHz"),
        ("Analysis band", f"{LOW_CUT / 1000:.0f}–{HIGH_CUT / 1000:.0f} kHz", "Band-pass filter"),
    ])

    max_t = round(duration, 2)
    lo, hi = st.slider("Display window (s)", 0.0, max_t, (0.0, min(1.0, max_t)), step=0.05)
    if hi - lo < 0.05:
        hi = min(max_t, lo + 0.05)
    i0, i1 = int(lo * FS), min(n, int(hi * FS))
    st.caption(
        f"Grey bands mark the {EDGE_SAMPLES / FS:.2f} s at each end of the recording that is excluded "
        "from feature extraction and CNN inference. Use the toolbar on each plot to zoom."
    )

    plots.show(plots.signal_window_figure(
        signal, i0, i1, n, title="Raw vibration waveform",
        y_title=plots.AMPLITUDE_LABEL, color=SIGNAL_COLORS["raw"]))
    st.caption("The unprocessed drive-end accelerometer signal as recorded. Amplitude units are those stored in the file and are not rescaled.")

    if filtered is not None:
        plots.show(plots.signal_window_figure(
            filtered, i0, i1, n, title=f"Band-pass filtered waveform ({LOW_CUT / 1000:.0f}–{HIGH_CUT / 1000:.0f} kHz)",
            y_title=plots.AMPLITUDE_LABEL, color=SIGNAL_COLORS["filtered"]))
        st.caption("Fourth-order Butterworth band-pass filter (zero-phase). It isolates the high-frequency "
                   "resonance band in which bearing-defect impacts are most visible.")
    else:
        note("warning", "Filtered waveform unavailable: src/preprocessing.py does not expose the band-pass "
                        "filter separately. Apply the small patch described in the delivery notes to enable it.")

    plots.show(plots.signal_window_figure(
        envelope, i0, i1, n, title="Hilbert envelope of the filtered signal",
        y_title="Envelope amplitude (recorded units)", color=SIGNAL_COLORS["envelope"]))
    st.caption(f"The instantaneous amplitude of the filtered signal. Repeating impacts from a localised defect "
               f"appear as periodic bursts; the envelope dominant frequency feature is taken from the "
               f"0–{ENVELOPE_MAX_FREQ} Hz spectrum of this curve.")


# ------------------------------------------------------------ diagnosis

def render_diagnosis(analysis) -> None:
    if analysis is None:
        st.info(EMPTY_NO_ANALYSIS)
        return

    summary = summarise(analysis.results)
    section("Diagnostic summary", f"Recording: {analysis.recording.name}")
    banner(summary.tone, summary.title, summary.detail)

    section("Model predictions")
    for col, result in zip(st.columns(3), analysis.results):
        col.markdown(model_card(result), unsafe_allow_html=True)

    st.write("")
    note("neutral",
         "Probabilities are raw model outputs for this recording and are not calibrated. A probability is "
         "not an accuracy: it does not say how often the model is correct. The condition is assigned "
         "by comparing the Inner Race fault probability with 50%.")
    if analysis.segment_count is not None:
        st.caption(f"Complete one-second segments analysed: {analysis.segment_count}. "
                   "Inference time covers the model call only, measured once on this machine; it excludes "
                   "model loading and signal processing and can include one-off start-up cost.")
    for result in analysis.results:
        if not result.ok:
            st.warning(f"{result.spec.name} could not produce a result: {result.error}")


# ----------------------------------------------------------- comparison

def render_comparison(analysis) -> None:
    if analysis is None:
        st.info(EMPTY_NO_ANALYSIS)
        return

    section("Model comparison for this recording",
            "How the three models assessed the loaded recording. These are predictions, not performance metrics.")
    rows = []
    for r in analysis.results:
        ok = r.ok
        rows.append({
            "Model": r.spec.name,
            "Input": r.spec.input_repr,
            "Aggregation over segments": r.spec.aggregation,
            "Predicted condition": r.condition if ok else "Unavailable",
            "Inner Race probability (%)": round(r.probability * 100, 2) if ok else None,
            "Healthy probability (%)": round(100 - r.probability * 100, 2) if ok else None,
            "Inference time (ms)": round(r.inference_ms, 3) if ok and r.inference_ms is not None else None,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if not any(r.ok for r in analysis.results):
        return
    left, right = st.columns(2)
    with left:
        plots.show(plots.probability_figure(analysis.results))
    with right:
        plots.show(plots.inference_time_figure(analysis.results))
    st.caption("Bars are coloured by predicted condition (green: Healthy, red: Inner Race Fault).")

    cnn = next((r for r in analysis.results if r.spec.key == "cnn" and r.ok), None)
    if cnn is not None and cnn.segment_probabilities is not None:
        plots.show(plots.cnn_segments_figure(cnn))
        st.caption("The CNN scores every one-second segment separately; the recording-level probability is "
                   "their mean. Spread between segments indicates how consistent the evidence is. "
                   "Logistic Regression and Random Forest classify one median feature vector, so they have no per-segment scores.")

    note("neutral",
         "Prediction versus performance: this page compares what the models said about one recording. "
         "How well each model performs in general is a separate question answered by the offline "
         "evaluation on the Overview tab.")


# ------------------------------------------------------------- features

FEATURE_INFO = (
    ("rms", "RMS", "recorded units",
     "Overall vibration energy of a segment. Energy often rises as a defect develops, but the absolute "
     "level also depends on load, speed and sensor placement, so it is interpreted by the trained model "
     "rather than against a fixed limit."),
    ("crest_factor", "Crest factor", "dimensionless",
     "Peak amplitude divided by RMS. High values indicate sparse, sharp impacts such as those from a "
     "localised raceway defect; smooth, steady vibration gives lower values."),
    ("envelope_dominant_frequency", "Envelope dominant frequency", "Hz",
     f"The strongest component (up to {ENVELOPE_MAX_FREQ} Hz, excluding DC) in the spectrum of the "
     f"{LOW_CUT / 1000:.0f}–{HIGH_CUT / 1000:.0f} kHz envelope. Periodic impacts modulate the envelope at the "
     "defect repetition rate. This tool reports the peak but does not compare it with bearing-specific defect frequencies."),
)


def render_features(analysis) -> None:
    if analysis is None:
        st.info(EMPTY_NO_ANALYSIS)
        return
    if analysis.recording_features is None:
        st.warning(f"Features could not be extracted: {analysis.feature_error}")
        return

    section("Traditional features",
            "Recording-level values (median across segments) supplied to Logistic Regression and Random Forest.")
    row = analysis.recording_features.iloc[0]
    metric_cards([(label, f"{row[key]:.4g}", unit) for key, label, unit, _ in FEATURE_INFO])

    st.write("")
    for _key, label, _unit, text in FEATURE_INFO:
        st.markdown(f"**{label}.** {text}")

    plots.show(plots.feature_segments_figure(analysis.segment_features))
    with st.expander("View feature tables"):
        st.write("Recording-level features (model input)")
        st.dataframe(analysis.recording_features, width="stretch", hide_index=True)
        st.write(f"Per-segment features ({analysis.segment_count} segments)")
        table = analysis.segment_features.copy()
        table.insert(0, "segment", range(1, len(table) + 1))
        st.dataframe(table, width="stretch", hide_index=True)