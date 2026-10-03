from ucimlrepo import fetch_ucirepo
from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_ID = 891

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_FILE = DATA_DIR / "diabetes_health_indicators.csv"


# ---------------------------------------------------------
# Create data directory if it does not exist
# ---------------------------------------------------------

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Download dataset from UCI
# ---------------------------------------------------------

print("=" * 70)
print("DOWNLOADING DIABETES HEALTH INDICATORS DATASET")
print("=" * 70)

print("\nFetching UCI dataset ID:", DATASET_ID)

dataset = fetch_ucirepo(id=DATASET_ID)


# ---------------------------------------------------------
# Extract features and target
# ---------------------------------------------------------

X = dataset.data.features
y = dataset.data.targets


print("\nDataset downloaded successfully.")


# ---------------------------------------------------------
# Combine features and target
# ---------------------------------------------------------

df = pd.concat([X, y], axis=1)


# ---------------------------------------------------------
# Save as CSV
# ---------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)


# ---------------------------------------------------------
# Basic verification
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET VERIFICATION")
print("=" * 70)

print("\nSaved file:")
print(OUTPUT_FILE)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nTarget distribution:")

if "Diabetes_binary" in df.columns:
    print(df["Diabetes_binary"].value_counts())
    print("\nTarget percentages:")
    print(df["Diabetes_binary"].value_counts(normalize=True).mul(100).round(2))

else:
    print("WARNING: Diabetes_binary column was not found.")


print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)