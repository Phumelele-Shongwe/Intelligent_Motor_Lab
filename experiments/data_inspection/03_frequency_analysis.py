from pathlib import Path
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, welch, detrend

# Dataset directory
DATASET_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab\dataset"
)

# Recording paths
healthy_path = DATASET_ROOT / "Normal" / "97_Normal_0.mat"

faulty_path = (
    DATASET_ROOT
    / "12k_Drive_End_Bearing_Fault_Data"
    / "IR" / "007" / "105_0.mat"
)

# Load MATLAB files
healthy_data = loadmat(healthy_path)
faulty_data = loadmat(faulty_path)

# Extract drive-end signals
healthy_signal = healthy_data["X097_DE_time"].ravel()
faulty_signal = faulty_data["X105_DE_time"].ravel()

# Sampling frequency confirmed by repository documentation
FS = 12000

print("Healthy samples:", len(healthy_signal))
print("Faulty samples:", len(faulty_signal))
print("Sampling frequency:", FS, "Hz")

#Step 2: Extract Equal Length Signals

# Select one second of data from each recording
N = 12000

healthy_segment = healthy_signal[:N]
faulty_segment = faulty_signal[:N]

# Remove the DC component / mean
healthy_segment = detrend(healthy_segment, type="constant")
faulty_segment = detrend(faulty_segment, type="constant")

print("Healthy segment:", healthy_segment.shape)
print("Faulty segment:", faulty_segment.shape)

print("Segment duration:", N / FS, "seconds")
print("Frequency resolution:", FS / N, "Hz")

#Step 3: Apply Hann Window

# Generate Hann window
window = windows.hann(N, sym=True)

# Apply window to each signal
healthy_windowed = healthy_segment * window
faulty_windowed = faulty_segment * window

# Visualise the window
plt.figure(figsize=(12, 4))

plt.plot(window)

plt.title("Hann Window")
plt.xlabel("Sample index")
plt.ylabel("Window amplitude")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#Calculate the FFT

# Calculate real-valued FFT
healthy_fft = np.fft.rfft(healthy_windowed)
faulty_fft = np.fft.rfft(faulty_windowed)

# Generate frequency bins
frequencies = np.fft.rfftfreq(N, d=1/FS)

print("Number of frequency bins:", len(frequencies))
print("Frequency spacing:", frequencies[1] - frequencies[0], "Hz")
print("Maximum frequency:", frequencies[-1], "Hz")

# Calculate the amplitude spectrum
def calculate_amplitude_spectrum(fft_values, window):

    # Calculate magnitude
    amplitude = np.abs(fft_values)

    # Correct for window coherent gain
    amplitude = amplitude / np.sum(window)

    # Convert to one-sided amplitude spectrum
    if len(amplitude) > 2:
        amplitude[1:-1] *= 2

    return amplitude


healthy_amplitude = calculate_amplitude_spectrum(
    healthy_fft.copy(), window
)

faulty_amplitude = calculate_amplitude_spectrum(
    faulty_fft.copy(), window
)

#Frequency Spectra Plot

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# Healthy bearing
axes[0].plot(
    frequencies,
    healthy_amplitude,
    linewidth=0.8
)

axes[0].set_title("Healthy Bearing - Frequency Spectrum")
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("Amplitude (g)")
axes[0].set_xlim(0, FS / 2)
axes[0].grid(True, alpha=0.3)

# Inner-race fault
axes[1].plot(
    frequencies,
    faulty_amplitude,
    linewidth=0.8
)

axes[1].set_title("Inner-Race Fault - Frequency Spectrum")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Amplitude (g)")
axes[1].set_xlim(0, FS / 2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Calculate Power Spectral Density using Welch's method

f_healthy, psd_healthy = welch(
    healthy_signal,
    fs=FS,
    window="hann",
    nperseg=4096,
    noverlap=2048,
    detrend="constant",
    scaling="density"
)

f_faulty, psd_faulty = welch(
    faulty_signal,
    fs=FS,
    window="hann",
    nperseg=4096,
    noverlap=2048,
    detrend="constant",
    scaling="density"
)

# Plot PSD
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

axes[0].plot(f_healthy, psd_healthy)
axes[0].set_title("Healthy Bearing - Welch PSD")
axes[0].set_ylabel("PSD (g²/Hz)")
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_xlim(0, FS / 2)
axes[0].grid(True, alpha=0.3)

axes[1].plot(f_faulty, psd_faulty)
axes[1].set_title("Inner-Race Fault - Welch PSD")
axes[1].set_ylabel("PSD (g²/Hz)")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_xlim(0, FS / 2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



from scipy.signal import find_peaks

def strongest_peaks(frequencies, amplitude, count=5):

    # Ignore DC component
    peaks, _ = find_peaks(amplitude[1:])

    # Restore original frequency-bin indices
    peaks = peaks + 1

    # Sort by amplitude
    strongest = peaks[
        np.argsort(amplitude[peaks])[-count:][::-1]
    ]

    for index in strongest:
        print(
            f"Frequency: {frequencies[index]:8.2f} Hz | "
            f"Amplitude: {amplitude[index]:.6f} g"
        )


print("\nHEALTHY BEARING - STRONGEST PEAKS")
strongest_peaks(frequencies, healthy_amplitude)

print("\nINNER-RACE FAULT - STRONGEST PEAKS")
strongest_peaks(frequencies, faulty_amplitude)



##################Further Analysis: Zooming into the Healthy Spectrum Around 1066 Hz##################
# Zoom into the healthy spectrum around 1066 Hz

plt.figure(figsize=(12, 5))

plt.plot(
    frequencies,
    healthy_amplitude,
    linewidth=1
)

plt.xlim(950, 1180)

plt.title("Healthy Bearing - Spectrum Around 1066 Hz")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (g)")
plt.grid(True, alpha=0.3)

# Mark the observed peaks
for peak in [1036, 1066, 1096]:
    plt.axvline(peak, linestyle="--", alpha=0.6)
    plt.text(peak, 0.95 * max(healthy_amplitude[
        (frequencies >= 950) & (frequencies <= 1180)
    ]), str(peak), rotation=90, ha="right")

plt.tight_layout()
plt.show()


#Examine the faulty spectrum around its strongest peaks
plt.figure(figsize=(12, 5))

plt.plot(
    frequencies,
    faulty_amplitude,
    linewidth=1
)

plt.xlim(2200, 3800)

plt.title("Inner-Race Fault - Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (g)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#Compare the spectra on a logarithmic scale

plt.figure(figsize=(14, 6))

plt.semilogy(
    frequencies[1:],
    healthy_amplitude[1:],
    label="Healthy"
)

plt.semilogy(
    frequencies[1:],
    faulty_amplitude[1:],
    label="Inner-race fault"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (g), logarithmic scale")
plt.title("Healthy vs Inner-Race Fault - Frequency Comparison")

plt.xlim(0, 6000)
plt.grid(True, which="both", alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()