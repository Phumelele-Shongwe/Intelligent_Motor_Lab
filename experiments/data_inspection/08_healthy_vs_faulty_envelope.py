
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert, detrend

# ==================================================
# 1. DATASET CONFIGURATION
# ==================================================

ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab\dataset"
)

healthy_path = ROOT / "Normal" / "97_Normal_0.mat"

faulty_path = (
    ROOT
    / "12k_Drive_End_Bearing_Fault_Data"
    / "IR"
    / "007"
    / "105_0.mat"
)

FS = 12000
SEGMENT_LENGTH = FS  # One second
BPFI_ORDER = 5.4152

# ==================================================
# 2. LOAD RECORDINGS
# ==================================================

healthy_data = loadmat(healthy_path)
faulty_data = loadmat(faulty_path)

healthy = healthy_data["X097_DE_time"].ravel()
faulty = faulty_data["X105_DE_time"].ravel()

rpm_h = float(np.squeeze(healthy_data["X097RPM"]))
rpm_f = float(np.squeeze(faulty_data["X105RPM"]))

shaft_h = rpm_h / 60
shaft_f = rpm_f / 60

bpfi_h = BPFI_ORDER * shaft_h
bpfi_f = BPFI_ORDER * shaft_f

print("HEALTHY RPM:", rpm_h)
print("FAULTY RPM:", rpm_f)

print(f"Healthy BPFI: {bpfi_h:.3f} Hz")
print(f"Faulty BPFI: {bpfi_f:.3f} Hz")

# ==================================================
# 3. DESIGN BAND-PASS FILTER
# ==================================================

sos = butter(
    4,
    [2000, 5000],
    btype="bandpass",
    fs=FS,
    output="sos"
)

# ==================================================
# 4. PROCESS EACH RECORDING
# ==================================================

def calculate_mean_envelope_spectrum(signal):

    # Filter the complete recording
    filtered = sosfiltfilt(sos, signal)

    # Extract amplitude envelope
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    # Remove filter edge regions
    edge = int(0.25 * FS)
    envelope = envelope[edge:-edge]

    # Process complete one-second segments
    number_of_segments = len(envelope) // SEGMENT_LENGTH

    window = np.hanning(SEGMENT_LENGTH)

    spectra = []

    for i in range(number_of_segments):

        start = i * SEGMENT_LENGTH
        end = start + SEGMENT_LENGTH

        segment = envelope[start:end]

        # Remove segment DC component
        segment = detrend(segment, type="constant")

        # Apply Hann window
        segment = segment * window

        # FFT
        E = np.fft.rfft(segment)

        # One-sided amplitude spectrum
        amplitude = np.abs(E) / np.sum(window)
        amplitude[1:-1] *= 2

        spectra.append(amplitude)

    spectra = np.array(spectra)

    frequencies = np.fft.rfftfreq(
        SEGMENT_LENGTH,
        d=1 / FS
    )

    mean_spectrum = np.mean(spectra, axis=0)

    return frequencies, mean_spectrum, number_of_segments


freq_h, spec_h, segments_h = (
    calculate_mean_envelope_spectrum(healthy)
)

freq_f, spec_f, segments_f = (
    calculate_mean_envelope_spectrum(faulty)
)

print("\nNUMBER OF SEGMENTS")
print("Healthy:", segments_h)
print("Faulty:", segments_f)

# ==================================================
# 5. COMPARE SPECTRA
# ==================================================

plt.figure(figsize=(14, 6))

plt.plot(
    freq_h,
    spec_h,
    label="Healthy",
    alpha=0.85
)

plt.plot(
    freq_f,
    spec_f,
    label="Inner-race fault",
    alpha=0.85
)

plt.axvline(
    bpfi_h,
    linestyle="--",
    label=f"Healthy BPFI: {bpfi_h:.2f} Hz"
)

plt.axvline(
    bpfi_f,
    linestyle=":",
    label=f"Faulty BPFI: {bpfi_f:.2f} Hz"
)

plt.xlim(0, 500)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mean envelope amplitude")
plt.title("Healthy vs Inner-Race Fault: Envelope Spectrum")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==================================================
# 6. INVESTIGATE BPFI AND HARMONICS
# ==================================================

def local_peak(spectrum, frequencies, target, radius=3):

    mask = (
        (frequencies >= target - radius)
        & (frequencies <= target + radius)
    )

    indices = np.where(mask)[0]

    if len(indices) == 0:
        return np.nan, np.nan

    index = indices[np.argmax(spectrum[indices])]

    return frequencies[index], spectrum[index]


print("\nBPFI HARMONIC COMPARISON")

print(
    f"{'Harmonic':<12}"
    f"{'Healthy (g)':<18}"
    f"{'Faulty (g)':<18}"
)

for harmonic in range(1, 4):

    target_h = harmonic * bpfi_h
    target_f = harmonic * bpfi_f

    observed_h, amp_h = local_peak(
        spec_h, freq_h, target_h
    )

    observed_f, amp_f = local_peak(
        spec_f, freq_f, target_f
    )

    print(
        f"{harmonic:<12}"
        f"{amp_h:<18.6f}"
        f"{amp_f:<18.6f}"
    )

# ==================================================
# 7. INVESTIGATE ORDER TRACKING
# ==================================================

order_h = freq_h / shaft_h
order_f = freq_f / shaft_f

plt.figure(figsize=(14, 6))

plt.plot(
    order_h,
    spec_h,
    label="Healthy"
)

plt.plot(
    order_f,
    spec_f,
    label="Inner-race fault"
)

plt.axvline(
    BPFI_ORDER,
    linestyle="--",
    label=f"Theoretical BPFI order: {BPFI_ORDER}"
)

plt.xlim(0, 20)
plt.xlabel("Shaft orders")
plt.ylabel("Mean envelope amplitude")
plt.title("Envelope Spectrum in Shaft Orders")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()