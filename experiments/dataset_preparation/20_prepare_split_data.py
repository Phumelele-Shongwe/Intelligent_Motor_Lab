
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

FEATURE_DIR = RESULTS_DIR / "prepared_data"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = RESULTS_DIR / "recording_split.csv"

FS = 12000
WINDOW_SECONDS = 1.0
EDGE_SECONDS = 0.25

WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)
EDGE_SAMPLES = int(FS * EDGE_SECONDS)

LOW_CUT = 2000
HIGH_CUT = 5000
ENVELOPE_MAX_FREQ = 500

FILTER_ORDER = 4

# ============================================================
# SIGNAL PROCESSING
# ============================================================


def load_de_signal(filepath):

    mat = loadmat(filepath)

    # Explicit correction for the known duplicate-variable file.
    if filepath.name == "99_Normal_2.mat":
        variable = "X099_DE_time"

    else:
        candidates = [
            key for key in mat.keys()
            if key.endswith("_DE_time")
        ]

        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one DE_time variable in {filepath}. "
                f"Found: {candidates}"
            )

        variable = candidates[0]

    if variable not in mat:
        raise KeyError(
            f"Variable {variable} not found in {filepath}"
        )

    signal = np.asarray(
        mat[variable],
        dtype=np.float64
    ).ravel()

    if not np.all(np.isfinite(signal)):
        raise ValueError(f"Non-finite values in {filepath}")

    return signal


def extract_envelope(signal):

    sos = butter(
        FILTER_ORDER,
        [LOW_CUT, HIGH_CUT],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    filtered = sosfiltfilt(sos, signal)

    envelope = np.abs(hilbert(filtered))

    return envelope


def extract_segment_features(signal, envelope):

    rms = np.sqrt(np.mean(signal ** 2))

    crest_factor = (
        np.max(np.abs(signal)) / rms
        if rms > 0 else 0.0
    )

    envelope_centered = envelope - np.mean(envelope)

    window = np.hanning(len(envelope_centered))

    spectrum = np.abs(
        np.fft.rfft(envelope_centered * window)
    )

    frequencies = np.fft.rfftfreq(
        len(envelope_centered),
        d=1 / FS
    )

    valid = frequencies <= ENVELOPE_MAX_FREQ

    if not np.any(valid):
        dominant_frequency = 0.0
    else:
        dominant_frequency = frequencies[valid][
            np.argmax(spectrum[valid])
        ]

    return {
        "rms": rms,
        "crest_factor": crest_factor,
        "envelope_dominant_frequency": dominant_frequency
    }


# ============================================================
# PROCESS ONE RECORDING
# ============================================================

def process_recording(row):

    filepath = PROJECT_ROOT / row["filepath"]

    signal = load_de_signal(filepath)

    if len(signal) <= 2 * EDGE_SAMPLES:
        raise ValueError(
            f"Recording too short: {row['filename']}"
        )

    # Filter and calculate envelope on the complete recording.
    envelope = extract_envelope(signal)

    # Remove the edges.
    signal = signal[EDGE_SAMPLES:-EDGE_SAMPLES]
    envelope = envelope[EDGE_SAMPLES:-EDGE_SAMPLES]

    features = []
    segments = []

    n_segments = len(signal) // WINDOW_SAMPLES

    for i in range(n_segments):

        start = i * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES

        raw_segment = signal[start:end]
        envelope_segment = envelope[start:end]

        feature_row = extract_segment_features(
            raw_segment,
            envelope_segment
        )

        feature_row.update({
            "filename": row["filename"],
            "fault_type": row["fault_type"],
            "label": int(row["label"]),
            "segment": i
        })

        features.append(feature_row)
        segments.append(raw_segment.astype(np.float32))

    return features, segments


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    manifest = pd.read_csv(MANIFEST)

    required_columns = {
        "filename",
        "filepath",
        "fault_type",
        "label",
        "split"
    }

    missing = required_columns - set(manifest.columns)

    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")

    if manifest["filename"].duplicated().any():
        raise ValueError("Duplicate recording filenames in manifest.")

    if set(manifest["split"]) != {"train", "test"}:
        raise ValueError("Unexpected split values.")

    outputs = {}

    for split in ["train", "test"]:

        subset = manifest[
            manifest["split"] == split
        ]

        all_features = []
        all_segments = []
        all_labels = []
        all_recordings = []

        print(f"\nPROCESSING {split.upper()} DATA")
        print("=" * 60)

        for _, row in subset.iterrows():

            print(f"Processing: {row['filename']}")

            features, segments = process_recording(row)

            all_features.extend(features)

            all_segments.extend(segments)

            all_labels.extend(
                [int(row["label"])] * len(segments)
            )

            all_recordings.extend(
                [row["filename"]] * len(segments)
            )

        feature_df = pd.DataFrame(all_features)

        X = np.stack(all_segments).astype(np.float32)
        y = np.asarray(all_labels, dtype=np.int64)

        # Save segment-level traditional features.
        feature_file = (
            FEATURE_DIR / f"{split}_features.csv"
        )

        feature_df.to_csv(feature_file, index=False)

        # Save raw segments for CNN training/evaluation.
        segment_file = (
            FEATURE_DIR / f"{split}_segments.npz"
        )

        np.savez_compressed(
            segment_file,
            X=X,
            y=y,
            recordings=np.asarray(all_recordings)
        )

        outputs[split] = {
            "features": feature_df,
            "X": X,
            "y": y
        }

        print(f"\nRecordings: {len(subset)}")
        print(f"Segments: {len(X)}")
        print(f"Signal array shape: {X.shape}")
        print(f"Feature file: {feature_file}")
        print(f"Segment file: {segment_file}")

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    train_files = set(
        manifest.loc[
            manifest["split"] == "train", "filename"
        ]
    )

    test_files = set(
        manifest.loc[
            manifest["split"] == "test", "filename"
        ]
    )

    assert train_files.isdisjoint(test_files)

    train_recordings = set(
        outputs["train"]["features"]["filename"]
    )

    test_recordings = set(
        outputs["test"]["features"]["filename"]
    )

    assert train_recordings == train_files
    assert test_recordings == test_files

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    for split in ["train", "test"]:

        df = outputs[split]["features"]

        print(f"\n{split.upper()}")
        print(f"Segments: {len(df)}")
        print("\nClass distribution:")
        print(df.groupby("fault_type")["filename"].nunique())

    print("\nSUCCESS: Training and test recordings remain separate.")


if __name__ == "__main__":
    main()