from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "diabetes_binary_health_indicators_BRFSS2015.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "Diabetes_binary"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("DIABETES HEALTH INDICATORS - DATA PREPROCESSING")
print("=" * 80)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Original dataset shape:", df.shape)


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

print("\n" + "=" * 80)
print("1. DUPLICATE REMOVAL")
print("=" * 80)

original_rows = len(df)

duplicate_count = df.duplicated().sum()

print(f"\nDuplicate rows found: {duplicate_count:,}")

df = df.drop_duplicates().reset_index(drop=True)

remaining_rows = len(df)

print(f"Rows before removal: {original_rows:,}")
print(f"Rows after removal : {remaining_rows:,}")
print(f"Rows removed       : {original_rows - remaining_rows:,}")


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 80)
print("2. FEATURE / TARGET SEPARATION")
print("=" * 80)

X = df.drop(columns=[TARGET])

y = df[TARGET]

print("\nFeature shape:", X.shape)
print("Target shape :", y.shape)

print("\nFeature columns:")

for index, column in enumerate(X.columns, start=1):
    print(f"{index:2}. {column}")


# ============================================================
# FIRST SPLIT
# ============================================================
# 70% training
# 30% temporary set
#
# Stratification preserves the class distribution.

print("\n" + "=" * 80)
print("3. TRAIN / VALIDATION / TEST SPLIT")
print("=" * 80)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# SECOND SPLIT
# ============================================================
# Split temporary 30% into:
# 15% validation
# 15% test

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)


print("\nSplit sizes:")

print("Training   :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing    :", X_test.shape)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("4. CLASS DISTRIBUTION AFTER SPLITTING")
print("=" * 80)


def print_class_distribution(name, target):

    counts = target.value_counts().sort_index()

    percentages = (
        target.value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(f"\n{name}:")

    for class_value in counts.index:

        print(
            f"Class {class_value}: "
            f"{counts[class_value]:,} "
            f"({percentages[class_value]:.2f}%)"
        )


print_class_distribution("Training", y_train)

print_class_distribution("Validation", y_val)

print_class_distribution("Testing", y_test)


# ============================================================
# FEATURE SCALING
# ============================================================
#
# IMPORTANT:
# StandardScaler is FIT ONLY on training data.
#
# This prevents information from validation/test sets
# leaking into the training process.

print("\n" + "=" * 80)
print("5. FEATURE SCALING")
print("=" * 80)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)


print("\nStandardScaler fitted on training data only.")

print("Training scaled shape   :", X_train_scaled.shape)
print("Validation scaled shape:", X_val_scaled.shape)
print("Testing scaled shape   :", X_test_scaled.shape)


# ============================================================
# CONVERT BACK TO DATAFRAMES
# ============================================================

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X.columns
)

X_val_scaled = pd.DataFrame(
    X_val_scaled,
    columns=X.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)


# ============================================================
# ADD TARGET BACK
# ============================================================

train_processed = X_train_scaled.copy()

train_processed[TARGET] = y_train.to_numpy()

val_processed = X_val_scaled.copy()

val_processed[TARGET] = y_val.to_numpy()

test_processed = X_test_scaled.copy()

test_processed[TARGET] = y_test.to_numpy()


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

print("\n" + "=" * 80)
print("6. SAVING PROCESSED DATA")
print("=" * 80)

train_file = PROCESSED_DIR / "train.csv"

val_file = PROCESSED_DIR / "validation.csv"

test_file = PROCESSED_DIR / "test.csv"


train_processed.to_csv(
    train_file,
    index=False
)

val_processed.to_csv(
    val_file,
    index=False
)

test_processed.to_csv(
    test_file,
    index=False
)


print("\nSaved:")

print(train_file)

print(val_file)

print(test_file)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("7. FINAL PREPROCESSING VERIFICATION")
print("=" * 80)

print("\nTraining shape  :", train_processed.shape)

print("Validation shape:", val_processed.shape)

print("Testing shape   :", test_processed.shape)


print("\nMissing values:")

print(
    "Training   :",
    train_processed.isnull().sum().sum()
)

print(
    "Validation :",
    val_processed.isnull().sum().sum()
)

print(
    "Testing    :",
    test_processed.isnull().sum().sum()
)


print("\n" + "=" * 80)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 80)