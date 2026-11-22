import io

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert

# STEP 1: SIGNAL CONFIGURATION

FS = 12000

WINDOW_SAMPLES = 12000
EDGE_SAMPLES = 3000

LOW_CUT = 2000
HIGH_CUT = 5000

ENVELOPE_MAX_FREQ = 500

# STEP 2: LOAD MATLAB SIGNAL: Load a selected vibration signal from MATLAB file bytes.

def load_mat_signal(file_bytes, variable):

    mat_data = loadmat(io.BytesIO(file_bytes))

    if variable not in mat_data:
        raise ValueError(
            f"Signal variable '{variable}' was not found."
        )

    signal = np.asarray(
        mat_data[variable]
    ).squeeze()

    if signal.ndim != 1:
        raise ValueError(
            "Expected a one-dimensional vibration signal."
        )

    return signal.astype(np.float64)


# STEP 3: SIGNAL VALIDATION: Validate and standardize an input vibration signal.

def validate_signal(signal):
    signal = np.asarray(
        signal,
        dtype=np.float64
    ).squeeze()

    if signal.ndim != 1:
        raise ValueError(
            "Input must be a one-dimensional signal."
        )

    if not np.isfinite(signal).all():
        raise ValueError(
            "Signal contains missing or invalid values."
        )

    minimum_length = (
        2 * EDGE_SAMPLES + WINDOW_SAMPLES
    )

    if len(signal) < minimum_length:
        raise ValueError(
            "Signal is too short. At least 1.5 seconds "
            "of data is required."
        )

    return signal

# STEP 4: BAND-PASS FILTERING AND ENVELOPE EXTRACTION
# Apply the original 2–5 kHz Butterworth band-pass filter, then calculate the Hilbert envelope.

def bandpass_filter(signal):

    sos = butter(
        4,
        [LOW_CUT, HIGH_CUT],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    return sosfiltfilt(
        sos,
        signal
    )


def extract_envelope(signal):

    filtered = bandpass_filter(signal)

    analytic_signal = hilbert(filtered)

    envelope = np.abs(analytic_signal)

    return envelope

#STEP 5: CNN SEGMENT PREPARATION: Prepare raw vibration segments for CNN inference.

def prepare_cnn_segments(signal, normalization):

    # Remove 0.25 seconds from both edges.
    trimmed = signal[
        EDGE_SAMPLES:-EDGE_SAMPLES
    ]
    # Divide into non-overlapping 1-second segments.
    number_of_segments = (
        len(trimmed) // WINDOW_SAMPLES
    )

    segments = []

    for i in range(number_of_segments):

        start = i * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES

        segments.append(
            trimmed[start:end]
        )

    segments = np.asarray(
        segments,
        dtype=np.float32
    )
    
    # 3. Normalize using training-set mean and standard deviation.
    mean = float(normalization["mean"])
    std = float(normalization["std"])

    if std <= 0:
        raise ValueError(
            "Invalid CNN normalization parameters."
        )

    segments = (segments - mean) / std

    return segments