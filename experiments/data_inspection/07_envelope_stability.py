
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert, detrend

# Load recording
data = loadmat(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab\dataset"
    r"\12k_Drive_End_Bearing_Fault_Data\IR\007\105_0.mat"
)

x = data["X105_DE_time"].ravel()

fs = 12000
segment_length = 12000

# Band-pass filter
sos = butter(
    4,
    [2000, 5000],
    btype="bandpass",
    fs=fs,
    output="sos"
)

# Filter entire recording
filtered = sosfiltfilt(sos, x)

# Extract envelope
envelope = np.abs(hilbert(filtered))

# Remove overall mean
envelope = detrend(envelope, type="constant")

window = np.hanning(segment_length)

freq = np.fft.rfftfreq(segment_length, 1/fs)

spectra = []
peak_amplitudes = []

# Analyse non-overlapping one-second segments
number_of_segments = len(envelope) // segment_length

for i in range(number_of_segments):

    start = i * segment_length
    end = start + segment_length

    segment = envelope[start:end]

    spectrum = np.abs(
        np.fft.rfft(segment * window)
    ) / np.sum(window)

    spectrum[1:-1] *= 2

    spectra.append(spectrum)

    # Amplitude close to 162 Hz
    index = np.argmin(np.abs(freq - 162))

    peak_amplitudes.append(spectrum[index])

spectra = np.array(spectra)

print("Number of segments:", number_of_segments)

for i, amplitude in enumerate(peak_amplitudes):

    print(
        f"Segment {i+1}: "
        f"162 Hz amplitude = {amplitude:.6f}"
    )

# Plot stability
plt.figure(figsize=(10, 5))

plt.plot(
    range(1, number_of_segments + 1),
    peak_amplitudes,
    marker="o"
)

plt.xlabel("Segment number")
plt.ylabel("Amplitude near 162 Hz")
plt.title("Stability of the 162 Hz envelope component")
plt.grid(True)

plt.tight_layout()
plt.show()

# Average envelope spectrum
mean_spectrum = np.mean(spectra, axis=0)

plt.figure(figsize=(12, 5))

plt.plot(freq, mean_spectrum)

plt.xlim(0, 500)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mean amplitude (g)")
plt.title("Mean envelope spectrum across recording")

plt.grid(True)
plt.tight_layout()
plt.show()