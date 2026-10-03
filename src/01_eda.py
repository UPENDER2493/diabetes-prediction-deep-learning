from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "diabetes_binary_health_indicators_BRFSS2015.csv"
)

VISUALIZATION_DIR = PROJECT_ROOT / "visualizations"

VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("DIABETES HEALTH INDICATORS - EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")


# ============================================================
# 1. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("1. BASIC DATASET INFORMATION")
print("=" * 80)

print("\nDataset shape:")
print(df.shape)

print("\nNumber of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# ============================================================
# 2. COLUMN INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("2. COLUMN INFORMATION")
print("=" * 80)

print("\nColumn names:")

for index, column in enumerate(df.columns, start=1):
    print(f"{index:2}. {column}")


# ============================================================
# 3. DATA TYPES
# ============================================================

print("\n" + "=" * 80)
print("3. DATA TYPES")
print("=" * 80)

print(df.dtypes)


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("4. MISSING VALUE ANALYSIS")
print("=" * 80)

missing_values = df.isnull().sum()

print("\nMissing values per column:")
print(missing_values)

total_missing = missing_values.sum()

print("\nTotal missing values:", total_missing)


# ============================================================
# 5. DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("5. DUPLICATE ANALYSIS")
print("=" * 80)

duplicate_count = df.duplicated().sum()

duplicate_percentage = (
    duplicate_count / len(df)
) * 100

print("\nDuplicate rows:", duplicate_count)
print(f"Duplicate percentage: {duplicate_percentage:.2f}%")


# ============================================================
# 6. SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("6. SUMMARY STATISTICS")
print("=" * 80)

print(df.describe().T)


# ============================================================
# 7. TARGET DISTRIBUTION
# ============================================================

TARGET = "Diabetes_binary"

print("\n" + "=" * 80)
print("7. TARGET DISTRIBUTION")
print("=" * 80)

target_counts = df[TARGET].value_counts().sort_index()

target_percentages = (
    df[TARGET]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

target_summary = pd.DataFrame(
    {
        "Count": target_counts,
        "Percentage": target_percentages.round(2),
    }
)

print(target_summary)


# ============================================================
# TARGET DISTRIBUTION VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 6))

ax = sns.countplot(
    data=df,
    x=TARGET
)

plt.title("Diabetes Target Distribution")
plt.xlabel("Diabetes Binary")
plt.ylabel("Number of Records")

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "01_target_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. FEATURE DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 80)
print("8. FEATURE DISTRIBUTION ANALYSIS")
print("=" * 80)

feature_columns = [
    column
    for column in df.columns
    if column != TARGET
]

print("\nNumber of input features:", len(feature_columns))


# Plot distributions of all features

fig, axes = plt.subplots(
    5,
    5,
    figsize=(18, 18)
)

axes = axes.flatten()

for index, column in enumerate(feature_columns):

    sns.histplot(
        data=df,
        x=column,
        discrete=True,
        ax=axes[index]
    )

    axes[index].set_title(column)
    axes[index].set_xlabel("")
    axes[index].set_ylabel("")

# Hide unused axes

for index in range(len(feature_columns), len(axes)):
    axes[index].set_visible(False)

plt.suptitle(
    "Feature Distributions",
    fontsize=18,
    y=1.02
)

plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "02_feature_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 80)
print("9. CORRELATION ANALYSIS")
print("=" * 80)

correlation_matrix = df.corr(numeric_only=True)

target_correlations = (
    correlation_matrix[TARGET]
    .drop(TARGET)
    .sort_values(
        key=lambda values: values.abs(),
        ascending=False
    )
)

print("\nFeatures most correlated with the target:")
print(target_correlations)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(16, 13))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.3
)

plt.title(
    "Correlation Matrix - Diabetes Health Indicators",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "03_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. TOP TARGET CORRELATIONS
# ============================================================

top_correlations = target_correlations.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_correlations.values,
    y=top_correlations.index
)

plt.title(
    "Top 10 Feature Correlations with Diabetes_binary"
)

plt.xlabel("Correlation")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "04_top_target_correlations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. CLASS BALANCE VISUALIZATION
# ============================================================

class_percentages = (
    df[TARGET]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

plt.figure(figsize=(8, 6))

plt.pie(
    class_percentages.values,
    labels=[
        "No Diabetes",
        "Diabetes / Prediabetes"
    ],
    autopct="%1.2f%%",
    startangle=90
)

plt.title("Diabetes Class Distribution")

plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "05_class_distribution_pie.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. HEALTH INDICATOR VS TARGET
# ============================================================

selected_features = [
    "HighBP",
    "HighChol",
    "BMI",
    "GenHlth",
    "PhysHlth",
    "Age",
]

for feature in selected_features:

    if feature not in df.columns:
        continue

    plt.figure(figsize=(8, 6))

    sns.countplot(
        data=df,
        x=feature,
        hue=TARGET
    )

    plt.title(
        f"{feature} Distribution by Diabetes Status"
    )

    plt.xlabel(feature)
    plt.ylabel("Number of Records")

    plt.tight_layout()

    safe_name = feature.lower()

    plt.savefig(
        VISUALIZATION_DIR
        / f"06_{safe_name}_vs_diabetes.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 13. FINAL EDA SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("EDA SUMMARY")
print("=" * 80)

print("\nDataset:")
print(f"Rows              : {df.shape[0]:,}")
print(f"Columns           : {df.shape[1]}")
print(f"Input features    : {len(feature_columns)}")
print(f"Missing values    : {total_missing:,}")
print(f"Duplicate rows    : {duplicate_count:,}")
print(
    f"Duplicate percent : {duplicate_percentage:.2f}%"
)

print("\nTarget distribution:")

for class_value in target_counts.index:
    print(
        f"Class {class_value}: "
        f"{target_counts[class_value]:,} "
        f"({target_percentages[class_value]:.2f}%)"
    )

print("\nVisualizations saved to:")
print(VISUALIZATION_DIR)

print("\n" + "=" * 80)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 80)