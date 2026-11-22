
import numpy as np
import pandas as pd

from src.preprocessing import (
    FS,
    WINDOW_SAMPLES,
    EDGE_SAMPLES,
    ENVELOPE_MAX_FREQ,
    extract_envelope
)

# ENVELOPE DOMINANT FREQUENCY: Calculate the dominant frequency of the envelope spectrum.

def envelope_dominant_frequency(envelope):

    # Remove the envelope mean.
    x = envelope - np.mean(envelope)

    n = len(x)

    # Apply a Hann window.
    window = np.hanning(n)

    # Calculate the single-sided FFT amplitude spectrum.
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
        n,
        d=1 / FS
    )

    # Restrict the search to 0–500 Hz, excluding DC.
    mask = (
        (frequencies > 0)
        & (frequencies <= ENVELOPE_MAX_FREQ)
    )

    f = frequencies[mask]
    a = amplitude[mask]

    if len(f) == 0 or np.sum(a) == 0:
        raise ValueError(
            "Unable to calculate envelope spectrum."
        )

    # Return the frequency corresponding to the largest peak.
    return float(f[np.argmax(a)])


# TRADITIONAL FEATURE EXTRACTION: Extract recording-level traditional ML features.

def extract_traditional_features(signal):

    envelope = extract_envelope(signal)

    # Remove boundary regions.
    raw_trimmed = signal[
        EDGE_SAMPLES:-EDGE_SAMPLES
    ]

    envelope_trimmed = envelope[
        EDGE_SAMPLES:-EDGE_SAMPLES
    ]

    number_of_segments = (
        len(raw_trimmed) // WINDOW_SAMPLES
    )

    feature_rows = []

    for i in range(number_of_segments):

        start = i * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES

        raw_segment = raw_trimmed[start:end]

        envelope_segment = envelope_trimmed[
            start:end
        ]

        rms = np.sqrt(
            np.mean(raw_segment ** 2)
        )

        peak = np.max(
            np.abs(raw_segment)
        )

        crest_factor = (
            peak / rms if rms > 0 else 0
        )

        dominant_frequency = (
            envelope_dominant_frequency(
                envelope_segment
            )
        )

        feature_rows.append({
            "rms": rms,
            "crest_factor": crest_factor,
            "envelope_dominant_frequency": dominant_frequency
        })

    if not feature_rows:
        raise ValueError(
            "No complete one-second segments available."
        )

    segment_features = pd.DataFrame(
        feature_rows
    )

    recording_features = segment_features[
        [
            "rms",
            "crest_factor",
            "envelope_dominant_frequency"
        ]
    ].median().to_frame().T

    return (
        recording_features,
        segment_features,
        raw_trimmed,
        number_of_segments
    )