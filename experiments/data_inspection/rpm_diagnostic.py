
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab"
)

audit_file = PROJECT_ROOT / "Inspect_Data" / "recording_audit.csv"

df = pd.read_csv(audit_file)

print("\nRECORDINGS WITH MISSING RPM")
print("=" * 60)

missing = df[df["rpm"].isna()]

print(
    missing[["filename", "signal_variable"]]
    .to_string(index=False)
)

print("\nRECORDINGS WITH UNUSUAL CHANNEL COUNTS")
print("=" * 60)

channel_counts = (
    df.groupby("filename")["signal_variable"]
    .agg(list)
)

for filename, channels in channel_counts.items():

    if len(channels) != 2:
        print(filename, ":", channels)

print("\nDIAGNOSTIC COMPLETE")