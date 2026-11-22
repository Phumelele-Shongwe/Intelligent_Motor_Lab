
from pathlib import Path
import scipy.io

PROJECT_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab"
)

DATASET_ROOT = PROJECT_ROOT / "dataset"

files_to_check = [
    "Normal/98_Normal_1.mat",
    "Normal/99_Normal_2.mat",

    "12k_Drive_End_Bearing_Fault_Data/IR/028/3001_0.mat",
    "12k_Drive_End_Bearing_Fault_Data/IR/028/3002_1.mat",
    "12k_Drive_End_Bearing_Fault_Data/IR/028/3003_2.mat",
    "12k_Drive_End_Bearing_Fault_Data/IR/028/3004_3.mat",

    "12k_Drive_End_Bearing_Fault_Data/B/028/3005_0.mat",
    "12k_Drive_End_Bearing_Fault_Data/B/028/3006_1.mat",
    "12k_Drive_End_Bearing_Fault_Data/B/028/3007_2.mat",
    "12k_Drive_End_Bearing_Fault_Data/B/028/3008_3.mat",
]

print("\nTARGETED MATLAB METADATA INSPECTION")
print("=" * 75)

for relative_path in files_to_check:

    file_path = DATASET_ROOT / relative_path

    print(f"\nFILE: {relative_path}")
    print("-" * 75)

    variables = scipy.io.whosmat(file_path)

    for name, shape, data_type in variables:

        print(
            f"{name:25} "
            f"Shape: {str(shape):18} "
            f"Type: {data_type}"
        )

print("\nINSPECTION COMPLETE")