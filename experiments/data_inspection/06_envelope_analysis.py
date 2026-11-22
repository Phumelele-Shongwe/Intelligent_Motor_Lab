
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import (
    butter,
    sosfiltfilt,
    hilbert,
    detrend,
    find_peaks
)

# --------------------------------------------------
# 1. Load the faulty bearing recording
# --------------------------------------------------

data = loadmat(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab\dataset"
    r"\12k_Drive_End_Bearing_Fault_Data\IR\007\105_0.mat"
)

x = data["X105_DE_time"].ravel()

fs = 12000
rpm = float(np.squeeze(data["X105RPM"]))

shaft_frequency = rpm / 60

# One-second segment
N = 12000

x = detrend(x[:N], type="constant")

time = np.arange(N) / fs

print("Sampling frequency:", fs)
print("RPM:", rpm)
print("Shaft frequency:", shaft_frequency)

# --------------------------------------------------
# 2. Band-pass filter
# --------------------------------------------------

# Exploratory band, based on observed spectral peaks.
# We will investigate whether other bands are useful later.

low = 2000
high = 5000

sos = butter(
    4,
    [low, high],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered = sosfiltfilt(sos, x)

# --------------------------------------------------
# 3. Extract the amplitude envelope
# --------------------------------------------------

analytic_signal = hilbert(filtered)

envelope = np.abs(analytic_signal)

# Remove DC component
envelope = detrend(envelope, type="constant")

# --------------------------------------------------
# 4. Calculate envelope spectrum
# --------------------------------------------------

window = np.hanning(N)

E = np.fft.rfft(envelope * window)

frequencies = np.fft.rfftfreq(N, d=1/fs)

amplitude = np.abs(E) / np.sum(window)
amplitude[1:-1] *= 2

# --------------------------------------------------
# 5. Plot the results
# --------------------------------------------------

fig, axes = plt.subplots(4, 1, figsize=(13, 13))

# Original signal
axes[0].plot(time, x)
axes[0].set_title("Original vibration signal")
axes[0].set_ylabel("Acceleration (g)")
axes[0].set_xlim(0, 0.2)
axes[0].grid(True)

# Filtered signal
axes[1].plot(time, filtered)
axes[1].set_title("Band-pass filtered vibration")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlim(0, 0.2)
axes[1].grid(True)

# Envelope
axes[2].plot(time, envelope)
axes[2].set_title("Extracted vibration envelope")
axes[2].set_ylabel("Amplitude")
axes[2].set_xlim(0, 0.2)
axes[2].grid(True)

# Envelope spectrum
axes[3].plot(frequencies, amplitude)
axes[3].set_title("Envelope spectrum")
axes[3].set_xlabel("Frequency (Hz)")
axes[3].set_ylabel("Amplitude")
axes[3].set_xlim(0, 500)
axes[3].grid(True)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. Identify prominent envelope frequencies
# --------------------------------------------------

region = (frequencies >= 5) & (frequencies <= 500)

indices = np.where(region)[0]

peaks, properties = find_peaks(
    amplitude[indices],
    prominence=0.0001,
    distance=3
)

peak_indices = indices[peaks]

strongest = peak_indices[
    np.argsort(amplitude[peak_indices])[-15:][::-1]
]

print("\nStrongest envelope-spectrum peaks:")

for i in strongest:
    f = frequencies[i]
    order = f / shaft_frequency

    print(
        f"{f:8.2f} Hz | "
        f"{amplitude[i]:.6f} | "
        f"{order:.3f} shaft orders"
    )