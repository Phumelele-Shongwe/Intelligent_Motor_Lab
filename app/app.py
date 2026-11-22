"""Intelligent Motor Condition Monitoring and Bearing Fault Diagnosis.

Run from the project root:  streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):  # make `src` and `dashboard` importable
    if _path not in sys.path:
        sys.path.insert(0, _path)

st.set_page_config(
    page_title="Intelligent Motor Condition Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard import views  # noqa: E402
from dashboard.components import key_value_table, page_header  # noqa: E402
from dashboard.pipeline import (  # noqa: E402
    InputError,
    build_csv_recording,
    build_mat_recording,
    csv_numeric_columns,
    mat_signal_variables,
    run_analysis,
    signal_views,
)
from dashboard.theme import inject_css  # noqa: E402
from src.inference import load_models as load_model_artifacts  # noqa: E402
from src.preprocessing import FS  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models"
REFERENCE_FIGURE = PROJECT_ROOT / "results" / "figures" / "final_model_comparison.png"
REQUIRED_MODEL_FILES = (
    "logistic_regression.joblib", "random_forest.joblib", "cnn_normalization.json", "bearing_cnn.pth",
)

DEMO_RECORDINGS = {
    "Healthy demonstration": {
        "path": PROJECT_ROOT / "dataset" / "Normal" / "100_Normal_3.mat",
        "variable": "X100_DE_time",
    },
    "Inner Race demonstration": {
        "path": PROJECT_ROOT / "dataset" / "12k_Drive_End_Bearing_Fault_Data" / "IR" / "014" / "170_1.mat",
        "variable": "X170_DE_time",
    },
}

DEMO_MODE, UPLOAD_MODE = "Demonstration recording", "Upload recording"


# ------------------------------------------------------------- caching

@st.cache_resource(show_spinner=False)
def get_models():
    return load_model_artifacts(MODEL_DIR)


@st.cache_data(show_spinner=False)
def cached_mat_recording(file_bytes, name, variable, source):
    return build_mat_recording(file_bytes, name, variable, source)


@st.cache_data(show_spinner=False)
def cached_csv_recording(file_bytes, name, column):
    return build_csv_recording(file_bytes, name, column)


@st.cache_data(show_spinner=False)
def cached_signal_views(_signal, fingerprint):
    return signal_views(_signal)


# ------------------------------------------------------------- sidebar

def sidebar_input():
    """Render the data-source controls; return the loaded Recording or None."""
    recording = None
    with st.sidebar:
        st.markdown("#### Data source")
        mode = st.radio("Input source", [DEMO_MODE, UPLOAD_MODE], label_visibility="collapsed")

        try:
            if mode == DEMO_MODE:
                choice = st.selectbox("Demonstration recording", list(DEMO_RECORDINGS))
                info = DEMO_RECORDINGS[choice]
                if info["path"].exists():
                    recording = cached_mat_recording(
                        info["path"].read_bytes(), info["path"].name, info["variable"], "Demonstration"
                    )
                else:
                    st.error(f"Demonstration file not found: {info['path']}")
            else:
                upload = st.file_uploader(
                    "Upload a CWRU .mat recording", type=["mat", "csv"],
                    help="A .mat file with a Drive-End variable (name ending in _DE_time), "
                         "or a CSV with one numeric signal column.",
                )
                if upload is not None:
                    data, name = upload.getvalue(), upload.name
                    if name.lower().endswith(".mat"):
                        variables = mat_signal_variables(data)
                        variable = variables[0] if len(variables) == 1 else st.selectbox(
                            "Drive-End signal variable", variables)
                        recording = cached_mat_recording(data, name, variable, "Uploaded .mat")
                    else:
                        column = st.selectbox("Signal column", csv_numeric_columns(data))
                        recording = cached_csv_recording(data, name, column)
                else:
                    st.caption("No file selected.")
        except InputError as error:
            st.error(str(error))

        if recording is not None:
            st.success("Recording loaded")
            key_value_table([
                ("File", recording.name),
                ("Variable", recording.variable),
                ("Samples", f"{len(recording.signal):,}"),
                ("Duration", f"{len(recording.signal) / FS:.2f} s at {FS // 1000} kHz"),
            ])

        st.divider()
        run = st.button("Run bearing diagnosis", type="primary", width="stretch",
                        disabled=recording is None)
    return recording, run


# ---------------------------------------------------------------- main

def main():
    inject_css()
    page_header(
        "Intelligent Motor Condition Monitoring",
        "Bearing fault diagnosis from vibration recordings using Logistic Regression, Random Forest and a 1D CNN",
    )

    recording, run = sidebar_input()

    if run and recording is not None:
        try:
            with st.spinner("Loading models and analysing the vibration signal..."):
                models = get_models()
                st.session_state["analysis"] = run_analysis(recording, models)
        except FileNotFoundError as error:
            st.session_state.pop("analysis", None)
            st.error(f"A trained model file is missing: {error}. Expected in {MODEL_DIR}: "
                     + ", ".join(REQUIRED_MODEL_FILES))
        except Exception as error:
            st.session_state.pop("analysis", None)
            st.error(f"Diagnosis failed: {type(error).__name__}: {error}")

    analysis = st.session_state.get("analysis")
    if analysis is not None and (recording is None or analysis.recording.fingerprint != recording.fingerprint):
        analysis = None  # the input changed since the last run; do not show stale results
        st.session_state.pop("analysis", None)

    filtered = envelope = None
    if recording is not None:
        try:
            filtered, envelope = cached_signal_views(recording.signal, recording.fingerprint)
        except Exception as error:
            st.error(f"Signal processing failed: {type(error).__name__}: {error}")
            recording = None

    tabs = st.tabs(["Overview", "Signal analysis", "Diagnosis", "Model comparison", "Feature interpretation"])
    with tabs[0]:
        views.render_overview(REFERENCE_FIGURE)
    with tabs[1]:
        views.render_signal(recording, filtered, envelope)
    with tabs[2]:
        views.render_diagnosis(analysis)
    with tabs[3]:
        views.render_comparison(analysis)
    with tabs[4]:
        views.render_features(analysis)

    st.divider()
    st.caption("Intelligent Motor Laboratory. Binary bearing fault classification (Healthy / Inner Race Fault), "
               "CWRU Drive-End data, 12 kHz.")


main()