
import numpy as np
from itertools import combinations

# Peaks from your current FFT results
healthy_peaks = np.array([
    1036, 1066, 1096, 1006, 1048,
    1017, 1081, 1138, 1021, 1100
])

faulty_peaks = np.array([
    2617, 2778, 2880, 2455, 2940,
    2719, 2898, 2599, 2557, 2736
])

shaft_h = 1796 / 60
shaft_f = 1797 / 60


def analyse_spacing(name, peaks, shaft_frequency):

    print(f"\n{'='*55}")
    print(name)
    print(f"Shaft frequency: {shaft_frequency:.4f} Hz")
    print("="*55)

    results = []

    for a, b in combinations(sorted(peaks), 2):

        spacing = abs(b - a)
        order = spacing / shaft_frequency

        results.append((spacing, order, a, b))

    results.sort(key=lambda x: x[0])

    print("\nSmallest peak spacings:")

    for spacing, order, a, b in results[:15]:

        print(
            f"{a:7.1f} -> {b:7.1f} Hz | "
            f"Spacing: {spacing:6.1f} Hz | "
            f"Order: {order:.3f}"
        )


analyse_spacing(
    "HEALTHY BEARING",
    healthy_peaks,
    shaft_h
)

analyse_spacing(
    "INNER-RACE FAULT",
    faulty_peaks,
    shaft_f
)