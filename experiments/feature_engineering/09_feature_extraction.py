import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis, skew
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

FS = 12000
WINDOW_SECONDS = 1.0
EDGE_SECONDS = 0.25

LOW_CUT = 2000
HIGH_CUT = 5000

BPFI_LIMIT = 2.0
ENVELOPE_MAX_FREQ = 500


from pathlib import Path

# Absolute path to the dataset directory
DATASET_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab\dataset"
)

DATASETS = [
    {
        "name": "Healthy",
        "label": 0,
        "file": DATASET_ROOT / "Normal" / "97_Normal_0.mat",
        "variable": "X097_DE_time",
        "rpm": 1796.0,
        "bpfi": 162.095,
    },
    {
        "name": "Inner_Race_Fault",
        "label": 1,
        "file": (
            DATASET_ROOT
            / "12k_Drive_End_Bearing_Fault_Data"
            / "IR"
            / "007"
            / "105_0.mat"
        ),
        "variable": "X105_DE_time",
        "rpm": 1797.0,
        "bpfi": 162.185,
    },
]

# ============================================================
# 2. LOAD MATLAB SIGNAL
# ============================================================

def load_signal(file_path, variable):

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    mat_data = loadmat(file_path)

    if variable not in mat_data:
        raise KeyError(f"{variable} not found in {file_path}")

    signal = np.asarray(mat_data[variable]).squeeze()

    if signal.ndim != 1:
        raise ValueError("Expected a one-dimensional signal.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values.")

    return signal.astype(np.float64)


# ============================================================
# 3. TIME-DOMAIN FEATURES
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
        "kurtosis": kurtosis(x, fisher=True, bias=False),
        "skewness": skew(x, bias=False),
    }


# ============================================================
# 4. ENVELOPE EXTRACTION
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
# 5. ENVELOPE FREQUENCY SPECTRUM
# ============================================================

def envelope_spectrum(envelope):

    x = envelope - np.mean(envelope)

    n = len(x)

    window = np.hanning(n)

    spectrum = np.abs(np.fft.rfft(x * window))

    # One-sided amplitude spectrum.
    amplitude = 2.0 * spectrum / np.sum(window)

    amplitude[0] /= 2.0

    if n % 2 == 0:
        amplitude[-1] /= 2.0

    frequencies = np.fft.rfftfreq(n, d=1 / FS)

    return frequencies, amplitude


# ============================================================
# 6. SPECTRAL FEATURES
# ============================================================

def spectral_features(envelope, bpfi):

    frequencies, amplitude = envelope_spectrum(envelope)

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

    # Locate the maximum amplitude near each
    # theoretical BPFI harmonic.

    features = {}

    for harmonic in range(1, 4):

        target = harmonic * bpfi

        region = np.abs(frequencies - target) <= BPFI_LIMIT

        if not np.any(region):
            raise ValueError("No FFT bins in BPFI search region.")

        features[f"bpfi_h{harmonic}_amplitude"] = (
            np.max(amplitude[region])
        )

    features["envelope_dominant_frequency"] = dominant_frequency

    features["envelope_spectral_centroid"] = spectral_centroid

    # Envelope spectral energy in selected frequency bands.
    # These are exploratory bands, not established fault bands.

    for low, high in [(0, 100), (100, 200), (200, 350), (350, 500)]:

        band = (f >= low) & (f < high)

        features[f"envelope_band_{low}_{high}_energy"] = (
            np.sum(a[band] ** 2)
        )

    return features


# ============================================================
# 7. PROCESS ONE RECORDING
# ============================================================

def process_recording(dataset):

    signal = load_signal(
        dataset["file"],
        dataset["variable"]
    )

    rpm = dataset["rpm"]

    shaft_frequency = rpm / 60

    bpfi = dataset["bpfi"]

    bpfi_order = bpfi / shaft_frequency

    # Extract envelope from the entire recording.
    filtered, envelope = extract_envelope(signal)

    # Exclude filter/Hilbert boundary regions.
    edge = int(EDGE_SECONDS * FS)

    start = edge
    stop = len(signal) - edge

    signal = signal[start:stop]
    envelope = envelope[start:stop]

    window_size = int(WINDOW_SECONDS * FS)

    number_of_segments = len(signal) // window_size

    rows = []

    for i in range(number_of_segments):

        begin = i * window_size
        end = begin + window_size

        raw_segment = signal[begin:end]
        envelope_segment = envelope[begin:end]

        features = time_features(raw_segment)

        # Envelope features
        features["envelope_rms"] = np.sqrt(
            np.mean(envelope_segment ** 2)
        )

        features["envelope_kurtosis"] = kurtosis(
            envelope_segment,
            fisher=True,
            bias=False
        )

        features["envelope_peak"] = np.max(envelope_segment)

        # Frequency-domain and bearing-specific features
        features.update(
            spectral_features(envelope_segment, bpfi)
        )

        # Metadata
        features["recording"] = dataset["name"]
        features["segment"] = i + 1
        features["label"] = dataset["label"]
        features["rpm"] = rpm
        features["shaft_frequency"] = shaft_frequency
        features["bpfi_order"] = bpfi_order

        rows.append(features)

    return rows


# ============================================================
# 8. MAIN EXPERIMENT
# ============================================================

def main():

    all_rows = []

    for dataset in DATASETS:

        print("\nProcessing:", dataset["name"])

        rows = process_recording(dataset)

        all_rows.extend(rows)

        print("Segments extracted:", len(rows))

    df = pd.DataFrame(all_rows)

    # Save the extracted feature dataset.
    output_file = Path("09_motor_features.csv")

    df.to_csv(output_file, index=False)

    print("\nFEATURE DATASET")
    print("=" * 60)

    print("Total segments:", len(df))
    print("Number of features:", len(df.columns) - 6)

    print("\nCLASS DISTRIBUTION")
    print(df["recording"].value_counts())

    print("\nFEATURE MEANS BY CLASS")

    numeric_features = df.select_dtypes(
        include=np.number
    ).columns.drop(["label", "segment"])

    print(
        df.groupby("recording")[numeric_features]
        .mean()
        .T
        .to_string()
    )

    print("\nSaved:", output_file)

    # Visualise selected features.
    selected = [
        "rms",
        "crest_factor",
        "kurtosis",
        "envelope_rms",
        "bpfi_h1_amplitude",
    ]

    fig, axes = plt.subplots(
        len(selected), 1,
        figsize=(10, 14)
    )

    for ax, feature in zip(axes, selected):

        for name, group in df.groupby("recording"):

            ax.plot(
                group["segment"],
                group[feature],
                marker="o",
                label=name
            )

        ax.set_title(feature)
        ax.set_xlabel("Segment")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("09_feature_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()