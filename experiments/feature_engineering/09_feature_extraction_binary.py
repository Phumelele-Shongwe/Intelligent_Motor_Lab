
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis, skew


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab"
)

DATASET_ROOT = PROJECT_ROOT / "dataset"

HEALTHY_ROOT = DATASET_ROOT / "Normal"
IR_ROOT = DATASET_ROOT / "12k_Drive_End_Bearing_Fault_Data" / "IR"

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "09_motor_features_binary.csv"

# Sampling-frequency assumption: 12 kHz dataset.
# Confirm this against the dataset documentation.
FS = 12000

WINDOW_SECONDS = 1.0
EDGE_SECONDS = 0.25

LOW_CUT = 2000
HIGH_CUT = 5000

ENVELOPE_MAX_FREQ = 500

# ============================================================
# 2. RECORDING DISCOVERY
# ============================================================

def discover_recordings():

    recordings = []

    # Healthy recordings.
    # Explicitly select X099 from 99_Normal_2.mat.
    healthy_files = [
        ("97_Normal_0.mat", "X097_DE_time"),
        ("98_Normal_1.mat", "X098_DE_time"),
        ("99_Normal_2.mat", "X099_DE_time"),
        ("100_Normal_3.mat", "X100_DE_time"),
    ]

    for filename, variable in healthy_files:

        recordings.append({
            "fault_type": "Healthy",
            "label": 0,
            "file": HEALTHY_ROOT / filename,
            "variable": variable,
        })

    # All 16 Inner Race recordings.
    ir_files = sorted(IR_ROOT.rglob("*.mat"))

    for file_path in ir_files:

        recordings.append({
            "fault_type": "Inner_Race",
            "label": 1,
            "file": file_path,
            "variable": None,
        })

    healthy_count = sum(
        item["label"] == 0 for item in recordings
    )

    ir_count = sum(
        item["label"] == 1 for item in recordings
    )

    print("\nRECORDING DISCOVERY")
    print("=" * 60)
    print("Healthy recordings:", healthy_count)
    print("Inner Race recordings:", ir_count)
    print("Total recordings:", len(recordings))

    if healthy_count != 4 or ir_count != 16:
        raise ValueError(
            "Expected 4 healthy and 16 IR recordings. "
            "Check the dataset directory."
        )

    return recordings


# ============================================================
# 3. LOAD SIGNAL AND RPM METADATA
# ============================================================


def extract_rpm(mat_data):

    rpm_variables = [
        key for key in mat_data
        if key.upper().endswith("RPM")
    ]

    if not rpm_variables:
        return np.nan

    values = np.asarray(
        mat_data[rpm_variables[0]]
    ).squeeze()

    if values.size != 1:
        return np.nan

    rpm = float(values)

    if not np.isfinite(rpm):
        return np.nan

    return rpm


def load_recording(dataset):

    file_path = dataset["file"]

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    mat_data = loadmat(file_path)

    variable = dataset["variable"]

    # Automatically discover DE channel for IR recordings.
    if variable is None:

        candidates = [
            key for key in mat_data
            if key.endswith("_DE_time")
        ]

        if len(candidates) != 1:
            raise ValueError(
                f"Expected one DE channel in {file_path}, "
                f"found: {candidates}"
            )

        variable = candidates[0]

    if variable not in mat_data:
        raise KeyError(
            f"{variable} not found in {file_path}"
        )

    signal = np.asarray(
        mat_data[variable]
    ).squeeze()

    if signal.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional signal: {file_path}"
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            f"Signal contains invalid values: {file_path}"
        )

    rpm = extract_rpm(mat_data)

    return signal.astype(np.float64), variable, rpm


# ============================================================
# 4. TIME-DOMAIN FEATURES
# ============================================================

def time_features(x):

    rms = np.sqrt(np.mean(x ** 2))
    std = np.std(x)

    peak = np.max(np.abs(x))
    peak_to_peak = np.ptp(x)

    crest_factor = peak / rms if rms > 0 else 0

    return {
        "mean": np.mean(x),
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurtosis(
            x, fisher=True, bias=False
        ),
        "skewness": skew(
            x, bias=False
        ),
    }


# ============================================================
# 5. ENVELOPE EXTRACTION
# ============================================================

def extract_envelope(signal):

    sos = butter(
        4,
        [LOW_CUT, HIGH_CUT],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    filtered = sosfiltfilt(sos, signal)

    analytic_signal = hilbert(filtered)

    envelope = np.abs(analytic_signal)

    return filtered, envelope


# ============================================================
# 6. ENVELOPE FREQUENCY SPECTRUM
# ============================================================

def envelope_spectrum(envelope):

    x = envelope - np.mean(envelope)

    n = len(x)

    window = np.hanning(n)

    spectrum = np.abs(
        np.fft.rfft(x * window)
    )

    amplitude = (
        2.0 * spectrum / np.sum(window)
    )

    amplitude[0] /= 2.0

    if n % 2 == 0:
        amplitude[-1] /= 2.0

    frequencies = np.fft.rfftfreq(
        n, d=1 / FS
    )

    return frequencies, amplitude


# ============================================================
# 7. SPECTRAL FEATURES
# ============================================================

def spectral_features(envelope):

    frequencies, amplitude = envelope_spectrum(
        envelope
    )

    mask = (
        (frequencies > 0) &
        (frequencies <= ENVELOPE_MAX_FREQ)
    )

    f = frequencies[mask]
    a = amplitude[mask]

    dominant_frequency = f[np.argmax(a)]

    spectral_centroid = (
        np.sum(f * a) / np.sum(a)
        if np.sum(a) > 0 else 0
    )

    features = {
        "envelope_dominant_frequency":
            dominant_frequency,

        "envelope_spectral_centroid":
            spectral_centroid,
    }

    # Exploratory envelope spectral energy bands.
    for low, high in [
        (0, 100),
        (100, 200),
        (200, 350),
        (350, 500),
    ]:

        band = (f >= low) & (f < high)

        features[
            f"envelope_band_{low}_{high}_energy"
        ] = np.sum(a[band] ** 2)

    return features


# ============================================================
# 8. PROCESS ONE RECORDING
# ============================================================

def process_recording(dataset):

    signal, variable, rpm = load_recording(dataset)

    filtered, envelope = extract_envelope(signal)

    # Exclude filter/Hilbert boundary regions.
    edge = int(EDGE_SECONDS * FS)

    start = edge
    stop = len(signal) - edge

    signal = signal[start:stop]
    envelope = envelope[start:stop]

    window_size = int(WINDOW_SECONDS * FS)

    number_of_segments = (
        len(signal) // window_size
    )

    rows = []

    relative_path = dataset["file"].relative_to(
        DATASET_ROOT
    )

    recording_id = str(relative_path)

    for i in range(number_of_segments):

        begin = i * window_size
        end = begin + window_size

        raw_segment = signal[begin:end]
        envelope_segment = envelope[begin:end]

        features = time_features(raw_segment)

        # Envelope statistical features.
        features["envelope_rms"] = np.sqrt(
            np.mean(envelope_segment ** 2)
        )

        features["envelope_kurtosis"] = kurtosis(
            envelope_segment,
            fisher=True,
            bias=False
        )

        features["envelope_peak"] = np.max(
            envelope_segment
        )

        # Envelope frequency-domain features.
        features.update(
            spectral_features(envelope_segment)
        )

        # Metadata: retained for analysis and grouping,
        # not intended as model predictors.
        features["recording_id"] = recording_id
        features["signal_variable"] = variable
        features["segment"] = i + 1
        features["fault_type"] = dataset["fault_type"]
        features["label"] = dataset["label"]
        features["rpm"] = rpm

        rows.append(features)

    return rows


# ============================================================
# 9. MAIN EXPERIMENT
# ============================================================

def main():

    datasets = discover_recordings()

    all_rows = []

    for index, dataset in enumerate(datasets, start=1):

        print(
            f"\n[{index}/{len(datasets)}] "
            f"Processing: {dataset['file'].name}"
        )

        rows = process_recording(dataset)

        all_rows.extend(rows)

        print("Segments extracted:", len(rows))

    df = pd.DataFrame(all_rows)

    # Arrange metadata columns first.
    metadata_columns = [
        "recording_id",
        "signal_variable",
        "segment",
        "fault_type",
        "label",
        "rpm",
    ]

    feature_columns = [
        column for column in df.columns
        if column not in metadata_columns
    ]

    df = df[metadata_columns + feature_columns]

    # Save consolidated feature dataset.
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nFEATURE DATASET")
    print("=" * 60)

    print("Total segments:", len(df))
    print("Unique recordings:", df["recording_id"].nunique())
    print("Number of candidate features:", len(feature_columns))

    print("\nCLASS DISTRIBUTION")
    print(
        df.groupby("fault_type")["label"]
        .agg(["count", "nunique"])
    )

    print("\nSEGMENTS PER RECORDING")
    print(
        df.groupby(
            ["fault_type", "recording_id"]
        ).size().to_string()
    )

    print("\nRPM METADATA")
    print(
        "Recordings with RPM:",
        df.groupby("recording_id")["rpm"].first().notna().sum()
    )

    print(
        "Recordings without RPM:",
        df.groupby("recording_id")["rpm"].first().isna().sum()
    )

    print("\nFEATURE MEANS BY CLASS")
    print(
        df.groupby("fault_type")[feature_columns]
        .mean()
        .T
        .to_string()
    )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()