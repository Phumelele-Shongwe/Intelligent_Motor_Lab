"""Input loading and analysis orchestration.

This module contains no Streamlit code. Signal validation, preprocessing,
feature extraction and model inference are delegated to ``src/``; nothing
is re-implemented here.
"""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import whosmat

from src.feature_extraction import extract_traditional_features
from src.inference import predict_cnn, predict_traditional
from src.preprocessing import (
    extract_envelope,
    load_mat_signal,
    prepare_cnn_segments,
    validate_signal,
)

try:  # exposed by the minimal preprocessing.py patch (see delivery notes)
    from src.preprocessing import bandpass_filter
except ImportError:  # pragma: no cover - older src/preprocessing.py
    bandpass_filter = None

# Same decision rule as the original application: probability >= 0.5.
PREDICTION_THRESHOLD = 0.5
CLASS_HEALTHY = "Healthy"
CLASS_FAULT = "Inner Race Fault"


class InputError(ValueError):
    """A user-facing problem with the supplied recording."""


@dataclass(frozen=True)
class ModelSpec:
    key: str          # key in the dict returned by src.inference.load_models
    name: str
    input_repr: str
    aggregation: str


MODEL_SPECS = (
    ModelSpec(
        "logistic_regression", "Logistic Regression", "Engineered features",
        "Median of per-segment features",
    ),
    ModelSpec(
        "random_forest", "Random Forest", "Engineered features",
        "Median of per-segment features",
    ),
    ModelSpec(
        "cnn", "1D CNN", "Raw vibration segments",
        "Mean of per-segment probabilities",
    ),
)


@dataclass(frozen=True)
class Recording:
    signal: np.ndarray
    name: str
    variable: str
    source: str
    fingerprint: str


@dataclass
class ModelResult:
    spec: ModelSpec
    probability: Optional[float] = None
    inference_ms: Optional[float] = None
    segment_probabilities: Optional[np.ndarray] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.probability is not None

    @property
    def prediction(self) -> int:
        return int(self.probability >= PREDICTION_THRESHOLD)

    @property
    def condition(self) -> str:
        return CLASS_FAULT if self.prediction == 1 else CLASS_HEALTHY


@dataclass
class Analysis:
    recording: Recording
    recording_features: Optional[pd.DataFrame]
    segment_features: Optional[pd.DataFrame]
    segment_count: Optional[int]
    feature_error: Optional[str]
    results: list


@dataclass(frozen=True)
class Summary:
    tone: str
    title: str
    detail: str


# ---------------------------------------------------------------- input

def _fingerprint(signal: np.ndarray, variable: str) -> str:
    return hashlib.sha1(signal.tobytes() + variable.encode()).hexdigest()[:16]


def mat_signal_variables(file_bytes: bytes) -> list:
    """List Drive-End vibration variables without loading the data."""
    try:
        entries = whosmat(io.BytesIO(file_bytes))
    except NotImplementedError as exc:
        raise InputError(
            "MATLAB v7.3 (HDF5) files are not supported. "
            "Re-save the file in MATLAB v7 format or earlier."
        ) from exc
    except Exception as exc:
        raise InputError("The file could not be read as a MATLAB .mat file.") from exc

    names = [name for name, _shape, _cls in entries if name.endswith("_DE_time")]
    if not names:
        found = ", ".join(n for n, _s, _c in entries) or "none"
        raise InputError(
            "No Drive-End vibration variable (name ending in '_DE_time') was found. "
            f"Variables in this file: {found}."
        )
    return names


def build_mat_recording(file_bytes: bytes, name: str, variable: str, source: str) -> Recording:
    try:
        signal = validate_signal(load_mat_signal(file_bytes, variable))
    except ValueError as exc:
        raise InputError(str(exc)) from exc
    return Recording(signal, name, variable, source, _fingerprint(signal, variable))


def csv_numeric_columns(file_bytes: bytes) -> list:
    try:
        frame = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise InputError("The file could not be read as a CSV table.") from exc
    columns = frame.select_dtypes(include=np.number).columns.tolist()
    if not columns:
        raise InputError("No numeric columns were found in the CSV file.")
    return columns


def build_csv_recording(file_bytes: bytes, name: str, column: str) -> Recording:
    try:
        frame = pd.read_csv(io.BytesIO(file_bytes))
        signal = validate_signal(frame[column].dropna().to_numpy(dtype=np.float64))
    except ValueError as exc:
        raise InputError(str(exc)) from exc
    return Recording(signal, name, column, "Uploaded CSV", _fingerprint(signal, column))


# ------------------------------------------------------------- analysis

def signal_views(signal: np.ndarray):
    """Band-pass filtered signal (or None if unavailable) and Hilbert envelope."""
    filtered = bandpass_filter(signal) if bandpass_filter is not None else None
    return filtered, extract_envelope(signal)


def run_analysis(recording: Recording, models: dict) -> Analysis:
    """Run all three existing models; one model failing does not hide the others."""
    signal = recording.signal

    features = segment_features = segment_count = feature_error = None
    try:
        features, segment_features, _raw, segment_count = extract_traditional_features(signal)
    except Exception as exc:
        feature_error = f"{type(exc).__name__}: {exc}"

    results = []
    for spec in MODEL_SPECS:
        result = ModelResult(spec)
        try:
            if spec.key == "cnn":
                segments = prepare_cnn_segments(signal, models["cnn_normalization"])
                start = time.perf_counter()
                probability, segment_probs = predict_cnn(
                    models["cnn"], segments, models["device"]
                )
                result.inference_ms = (time.perf_counter() - start) * 1000
                result.segment_probabilities = segment_probs
            else:
                if features is None:
                    raise RuntimeError(f"Feature extraction failed ({feature_error})")
                start = time.perf_counter()
                probability = predict_traditional(models[spec.key], features)
                result.inference_ms = (time.perf_counter() - start) * 1000
            result.probability = float(probability)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        results.append(result)

    return Analysis(recording, features, segment_features, segment_count, feature_error, results)


def summarise(results: list) -> Summary:
    """Describe agreement between the three models. This is not a fourth model."""
    ok = [r for r in results if r.ok]
    if not ok:
        return Summary("warning", "No diagnosis available", "None of the three models returned a result.")

    faults = [r for r in ok if r.prediction == 1]
    healthy = [r for r in ok if r.prediction == 0]
    n, total = len(ok), len(results)
    missing = f" Only {n} of {total} models returned a result." if n < total else ""

    if len(faults) == n:
        return Summary(
            "fault", "Inner Race Fault indicated",
            f"{n} of {n} models classify this recording as Inner Race Fault.{missing}",
        )
    if len(healthy) == n:
        return Summary(
            "healthy", "Healthy condition indicated",
            f"{n} of {n} models classify this recording as Healthy.{missing}",
        )
    return Summary(
        "warning", "Models disagree",
        f"{len(faults)} of {n} models indicate Inner Race Fault ("
        f"{', '.join(r.spec.name for r in faults)}) and {len(healthy)} indicate Healthy ("
        f"{', '.join(r.spec.name for r in healthy)}). Review the probabilities and signal "
        f"evidence before drawing a conclusion.{missing}",
    )