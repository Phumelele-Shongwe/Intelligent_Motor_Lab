
import sys
import json
import time

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

# Ensure project root is available for src imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import BearingCNN


# ============================================================
# 1. CONFIGURATION
# ============================================================

FEATURE_DIR = PROJECT_ROOT / "results" / "prepared_data"
MODEL_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "rms",
    "crest_factor",
    "envelope_dominant_frequency"
]

GROUP = "filename"
TARGET = "label"

DEVICE = torch.device("cpu")

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100

THRESHOLD = 0.5


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\nMODEL INFERENCE BENCHMARK")
print("=" * 70)

print("Device:", DEVICE)
print("Warm-up runs:", WARMUP_RUNS)
print("Benchmark repetitions:", BENCHMARK_RUNS)

train_features = pd.read_csv(
    FEATURE_DIR / "train_features.csv"
)

test_features = pd.read_csv(
    FEATURE_DIR / "test_features.csv"
)

test_data = np.load(
    FEATURE_DIR / "test_segments.npz"
)

X_test_cnn = test_data["X"].astype(np.float32)

test_recordings_cnn = (
    np.asarray(test_data["recordings"]).astype(str)
)

if len(test_recordings_cnn) != len(test_features):
    raise ValueError("CNN segments and feature rows are not aligned.")

if not np.array_equal(
    test_recordings_cnn,
    test_features[GROUP].astype(str).to_numpy()
):
    raise ValueError("Recording identifiers are not aligned.")

if not set(
    train_features[GROUP].astype(str).unique()
).isdisjoint(
    set(test_features[GROUP].astype(str).unique())
):
    raise ValueError("Recording leakage detected.")

# Median feature aggregation, exactly as in Script 23.
test_recordings = (
    test_features.groupby(GROUP)
    .agg(
        rms=("rms", "median"),
        crest_factor=("crest_factor", "median"),
        envelope_dominant_frequency=(
            "envelope_dominant_frequency",
            "median"
        ),
        label=(TARGET, "first")
    )
    .reset_index()
)

print("\nDATA VALIDATION")
print("-" * 70)
print("Test recordings:", len(test_recordings))
print("Test CNN segments:", len(X_test_cnn))


# ============================================================
# 3. LOAD TRADITIONAL MODELS
# ============================================================

print("\nLOADING TRADITIONAL MODELS")
print("-" * 70)

traditional_models = {}

for name in ["logistic_regression", "random_forest"]:

    path = MODEL_DIR / f"{name}.joblib"

    if not path.exists():
        raise FileNotFoundError(path)

    traditional_models[name] = joblib.load(path)

    print(f"[OK] {name} loaded")


# ============================================================
# 4. LOAD CNN AND TRAINING NORMALIZATION
# ============================================================

print("\nLOADING CNN")
print("-" * 70)

with open(
    MODEL_DIR / "cnn_normalization.json",
    "r"
) as file:

    normalization = json.load(file)

train_mean = float(normalization["mean"])
train_std = float(normalization["std"])

if not np.isfinite(train_mean):
    raise ValueError("Invalid training mean.")

if not np.isfinite(train_std) or train_std < 1e-8:
    raise ValueError("Invalid training standard deviation.")

checkpoint = torch.load(
    MODEL_DIR / "bearing_cnn.pth",
    map_location=DEVICE,
    weights_only=True
)

cnn_model = BearingCNN().to(DEVICE)

cnn_model.load_state_dict(
    checkpoint["model_state_dict"]
)

cnn_model.eval()

# Use the original training-set normalization.
X_test_normalized = (
    X_test_cnn - train_mean
) / train_std

X_test_tensor = torch.tensor(
    X_test_normalized[:, np.newaxis, :],
    dtype=torch.float32
)

print("[OK] CNN loaded")
print("[OK] Training normalization loaded")


# ============================================================
# 5. PREPARE RECORDING-LEVEL INPUTS
# ============================================================

recording_inputs = []

for _, row in test_recordings.iterrows():

    filename = str(row[GROUP])

    # Traditional model: one feature vector per recording.
    X_traditional = row[FEATURES].to_numpy(
        dtype=np.float64
    ).reshape(1, -1)

    # CNN: all segments belonging to this recording.
    indices = np.where(
        test_recordings_cnn == filename
    )[0]

    if len(indices) == 0:
        raise ValueError(
            f"No CNN segments found for {filename}"
        )

    X_cnn = X_test_tensor[indices]

    recording_inputs.append({
        "filename": filename,
        "traditional": X_traditional,
        "cnn": X_cnn,
        "segments": len(indices)
    })

print("\nRECORDING INPUTS")
print("-" * 70)

for item in recording_inputs:

    print(
        f"{item['filename']:<20} "
        f"CNN segments: {item['segments']}"
    )


# ============================================================
# 6. DEFINE INFERENCE FUNCTIONS
# ============================================================

def traditional_inference(model, X):

    probabilities = model.predict_proba(X)

    positive_index = list(model.classes_).index(1)

    probability = float(
        probabilities[0, positive_index]
    )

    prediction = int(probability >= THRESHOLD)

    return probability, prediction


def cnn_inference(X):

    with torch.inference_mode():

        logits = cnn_model(X)

        segment_probabilities = torch.sigmoid(logits)

        recording_probability = (
            segment_probabilities.mean().item()
        )

    prediction = int(
        recording_probability >= THRESHOLD
    )

    return recording_probability, prediction


# ============================================================
# 7. BENCHMARK FUNCTION
# ============================================================

def benchmark_inference(function, inputs):

    # Warm-up: exclude initial execution effects.
    for _ in range(WARMUP_RUNS):
        function(inputs)

    timings_ms = []

    # Repeated measurements.
    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        function(inputs)

        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000.0

        timings_ms.append(elapsed_ms)

    return timings_ms


# ============================================================
# 8. RUN BENCHMARK
# ============================================================

print("\nRUNNING INFERENCE BENCHMARK")
print("=" * 70)

benchmark_results = []

for model_name in [
    "logistic_regression",
    "random_forest",
    "cnn"
]:

    print(f"\nBenchmarking {model_name}...")

    model_timings = []

    for item in recording_inputs:

        if model_name == "cnn":

            function = cnn_inference
            inputs = item["cnn"]

        else:

            model = traditional_models[model_name]

            function = lambda X, m=model: (
                traditional_inference(m, X)
            )

            inputs = item["traditional"]

        timings = benchmark_inference(
            function,
            inputs
        )

        model_timings.extend(timings)

    timing_array = np.asarray(
        model_timings,
        dtype=np.float64
    )

    mean_latency = float(timing_array.mean())
    median_latency = float(np.median(timing_array))
    p95_latency = float(np.percentile(timing_array, 95))

    throughput = (
        1000.0 / mean_latency
        if mean_latency > 0 else np.nan
    )

    benchmark_results.append({

        "model": model_name,

        "recordings_benchmarked": len(recording_inputs),

        "segments_per_recording_mean": float(
            np.mean([
                item["segments"]
                for item in recording_inputs
            ])
        ),

        "warmup_runs_per_recording": WARMUP_RUNS,

        "benchmark_runs_per_recording": BENCHMARK_RUNS,

        "mean_latency_ms": mean_latency,

        "median_latency_ms": median_latency,

        "p95_latency_ms": p95_latency,

        "recordings_per_second": throughput
    })

    print("[OK] Completed")
    print(f"Mean latency:   {mean_latency:.4f} ms")
    print(f"Median latency: {median_latency:.4f} ms")
    print(f"P95 latency:    {p95_latency:.4f} ms")
    print(f"Throughput:     {throughput:.2f} recordings/s")


# ============================================================
# 9. SAVE BENCHMARK RESULTS
# ============================================================

benchmark_df = pd.DataFrame(benchmark_results)

output_path = (
    METRICS_DIR / "inference_benchmark.csv"
)

benchmark_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\nINFERENCE BENCHMARK SUMMARY")
print("=" * 70)

display_columns = [
    "model",
    "mean_latency_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "recordings_per_second"
]

print(
    benchmark_df[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSAVED RESULTS")
print("-" * 70)
print("Benchmark:", output_path)

print("\nINFERENCE BENCHMARK COMPLETED SUCCESSFULLY")
print("No model artifacts were modified.")