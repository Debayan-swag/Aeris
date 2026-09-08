import pandas as pd
from pathlib import Path

# Use absolute path to avoid OneDrive/directory issues
csv_path = Path(__file__).parent / "data" / "processed" / "metadata" / "metadata.csv"

df = pd.read_csv(csv_path)

print("=" * 70)
print("TERRAWATCH - METADATA.CSV INSPECTION")
print("=" * 70)
print(f"\nFile: {csv_path}")
print(f"Total rows: {len(df):,}")
print(f"\nColumns ({len(df.columns)}):")
print(df.columns.tolist())
print(f"\nFirst 5 rows:\n")
print(df.head().to_string())
print(f"\nData types:\n")
print(df.dtypes)
print(f"\nSummary statistics:\n")
print(df.describe())
print("\n" + "=" * 70)
