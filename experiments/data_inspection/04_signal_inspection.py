
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import find_peaks, detrend

# Dataset location
ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab\dataset"
)

healthy_data = loadmat(ROOT / "Normal" / "97_Normal_0.mat")

faulty_data = loadmat(
    ROOT / "12k_Drive_End_Bearing_Fault_Data"
    / "IR" / "007" / "105_0.mat"
)

healthy = healthy_data["X097_DE_time"].ravel()
faulty = faulty_data["X105_DE_time"].ravel()

# Documented rates for the selected 12 kHz recordings
FS_HEALTHY = 12000
FS_FAULTY = 12000

# Do not compare unless sampling rates match
if FS_HEALTHY != FS_FAULTY:
    raise ValueError("Sampling rates differ. Resample before comparison.")

FS = FS_HEALTHY
N = 12000

# Read RPM metadata
rpm_h = float(np.squeeze(healthy_data["X097RPM"]))
rpm_f = float(np.squeeze(faulty_data["X105RPM"]))

print("Healthy RPM:", rpm_h)
print("Faulty RPM:", rpm_f)
print("Healthy shaft frequency:", rpm_h / 60)
print("Faulty shaft frequency:", rpm_f / 60)

# Extract one-second segments
x_h = detrend(healthy[:N], type="constant")
x_f = detrend(faulty[:N], type="constant")

# Hann window
window = np.hanning(N)

# FFT
X_h = np.fft.rfft(x_h * window)
X_f = np.fft.rfft(x_f * window)

freq = np.fft.rfftfreq(N, d=1 / FS)

# One-sided amplitude spectrum
def amplitude_spectrum(X):
    A = np.abs(X) / np.sum(window)
    A[1:-1] *= 2
    return A

A_h = amplitude_spectrum(X_h)
A_f = amplitude_spectrum(X_f)

# Time axis
time = np.arange(N) / FS

# Plot time-domain signals and full spectra
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

axes[0, 0].plot(time, x_h)
axes[0, 0].set_xlim(0, 0.1)
axes[0, 0].set_title("Healthy - Time Domain")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Acceleration (g)")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(freq, A_h)
axes[0, 1].set_title("Healthy - Frequency Domain")
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].set_ylabel("Amplitude (g)")
axes[0, 1].set_xlim(0, 6000)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(time, x_f)
axes[1, 0].set_xlim(0, 0.1)
axes[1, 0].set_title("Inner-Race Fault - Time Domain")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Acceleration (g)")
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(freq, A_f)
axes[1, 1].set_title("Inner-Race Fault - Frequency Domain")
axes[1, 1].set_xlabel("Frequency (Hz)")
axes[1, 1].set_ylabel("Amplitude (g)")
axes[1, 1].set_xlim(0, 6000)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Inspect the suspicious frequency regions
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(freq, A_h)
axes[0].set_xlim(950, 1150)
axes[0].set_title("Healthy: 950-1150 Hz")
axes[0].set_ylabel("Amplitude (g)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(freq, A_f)
axes[1].set_xlim(2300, 3000)
axes[1].set_title("Faulty: 2300-3000 Hz")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Amplitude (g)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find prominent local peaks in the suspicious regions
def report_peaks(f, A, low, high, count=10):

    region = (f >= low) & (f <= high)
    indices = np.where(region)[0]

    local_peaks, _ = find_peaks(A[indices], distance=3)
    peak_indices = indices[local_peaks]

    strongest = peak_indices[
        np.argsort(A[peak_indices])[-count:][::-1]
    ]

    for i in strongest:
        print(f"{f[i]:.1f} Hz : {A[i]:.6f} g")


print("\nHealthy peaks around 1066 Hz:")
report_peaks(freq, A_h, 950, 1150)

print("\nFaulty peaks around 2300-3000 Hz:")
report_peaks(freq, A_f, 2300, 3000)