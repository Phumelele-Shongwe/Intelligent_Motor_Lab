
from pathlib import Path
import scipy.io
import numpy as np

PROJECT_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab"
)

NORMAL_ROOT = PROJECT_ROOT / "dataset" / "Normal"

file_98 = scipy.io.loadmat(
    NORMAL_ROOT / "98_Normal_1.mat"
)

file_99 = scipy.io.loadmat(
    NORMAL_ROOT / "99_Normal_2.mat"
)

print("\nHEALTHY RECORDING COMPARISON")
print("=" * 65)

# Compare X098 measurements across the two files.
for channel in ["DE", "FE"]:

    variable = f"X098_{channel}_time"

    signal_a = np.asarray(file_98[variable]).squeeze()
    signal_b = np.asarray(file_99[variable]).squeeze()

    print(f"\n{variable}")

    print("98 file samples:", len(signal_a))
    print("99 file samples:", len(signal_b))

    if signal_a.shape == signal_b.shape:

        print(
            "Exactly identical:",
            np.array_equal(signal_a, signal_b)
        )

        print(
            "Maximum absolute difference:",
            np.max(np.abs(signal_a - signal_b))
        )

    else:

        print("Different lengths: cannot compare element by element")


# Inspect the additional MATLAB variable.
print("\nADDITIONAL VARIABLE: ans")
print("-" * 65)

ans = np.asarray(file_99["ans"])

print("Shape:", ans.shape)
print("Value:", ans)
print("Data type:", ans.dtype)

print("\nCOMPARISON COMPLETE")