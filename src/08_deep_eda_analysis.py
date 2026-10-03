from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# WEEK 5 — DEEP EDA AND OUTLIER ANALYSIS
# ============================================================

DATA_PATH = Path(
    "data/diabetes_binary_health_indicators_BRFSS2015.csv"
)

OUTPUT_DIR = Path("visualizations")
REPORT_DIR = Path("models")

OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("DEEP EDA AND OUTLIER ANALYSIS")
print("=" * 80)

print(f"\nDataset shape: {df.shape}")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# 2. BASIC DATA QUALITY
# ============================================================

print("\n" + "=" * 80)
print("DATA QUALITY")
print("=" * 80)

missing_total = int(df.isna().sum().sum())
duplicate_total = int(df.duplicated().sum())

print(f"Missing values: {missing_total:,}")
print(f"Duplicate rows: {duplicate_total:,}")
print(
    f"Duplicate percentage: "
    f"{duplicate_total / len(df) * 100:.2f}%"
)

print("\nData types:")
print(df.dtypes.value_counts())


# ============================================================
# 3. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)

target_counts = df["Diabetes_binary"].value_counts().sort_index()
target_percentages = (
    df["Diabetes_binary"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

target_table = pd.DataFrame(
    {
        "Count": target_counts,
        "Percentage": target_percentages.round(2),
    }
)

print(target_table)

# Save target distribution
target_table.to_csv(
    REPORT_DIR / "deep_eda_target_distribution.csv"
)


# ============================================================
# 4. DUPLICATE IMPACT
# ============================================================

print("\n" + "=" * 80)
print("DUPLICATE IMPACT ANALYSIS")
print("=" * 80)

df_no_duplicates = df.drop_duplicates().copy()

before_counts = (
    df["Diabetes_binary"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

after_counts = (
    df_no_duplicates["Diabetes_binary"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

duplicate_impact = pd.DataFrame(
    {
        "Before_Duplicate_Removal_%": before_counts.round(2),
        "After_Duplicate_Removal_%": after_counts.round(2),
    }
)

duplicate_impact["Percentage_Point_Change"] = (
    duplicate_impact["After_Duplicate_Removal_%"]
    - duplicate_impact["Before_Duplicate_Removal_%"]
).round(2)

print(duplicate_impact)

duplicate_impact.to_csv(
    REPORT_DIR / "deep_eda_duplicate_impact.csv"
)


# ============================================================
# 5. NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("NUMERICAL SUMMARY")
print("=" * 80)

numeric_summary = df.describe().T

numeric_summary["missing"] = df.isna().sum()
numeric_summary["unique"] = df.nunique()

print(numeric_summary)

numeric_summary.to_csv(
    REPORT_DIR / "deep_eda_numeric_summary.csv"
)


# ============================================================
# 6. UNIQUE VALUES PER FEATURE
# ============================================================

print("\n" + "=" * 80)
print("UNIQUE VALUES PER FEATURE")
print("=" * 80)

unique_table = pd.DataFrame(
    {
        "Feature": df.columns,
        "Unique_Values": [
            df[column].nunique()
            for column in df.columns
        ],
    }
)

print(unique_table.to_string(index=False))

unique_table.to_csv(
    REPORT_DIR / "deep_eda_unique_values.csv",
    index=False
)


# ============================================================
# 7. IQR OUTLIER ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("IQR OUTLIER ANALYSIS")
print("=" * 80)

# Continuous / count-like variables for which an IQR
# investigation is meaningful.
outlier_features = [
    "BMI",
    "MentHlth",
    "PhysHlth",
]

outlier_records = []

for feature in outlier_features:

    series = df[feature].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (series < lower_bound)
        | (series > upper_bound)
    )

    outlier_count = int(outlier_mask.sum())

    outlier_percentage = (
        outlier_count / len(series) * 100
    )

    outlier_records.append(
        {
            "Feature": feature,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Outlier_Count": outlier_count,
            "Outlier_Percentage": outlier_percentage,
            "Minimum": series.min(),
            "Maximum": series.max(),
        }
    )

outlier_table = pd.DataFrame(outlier_records)

print(
    outlier_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)

outlier_table.to_csv(
    REPORT_DIR / "deep_eda_outlier_analysis.csv",
    index=False
)


# ============================================================
# 8. OUTLIER VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(16, 5)
)

for ax, feature in zip(axes, outlier_features):

    sns.boxplot(
        y=df[feature],
        ax=ax
    )

    ax.set_title(
        f"{feature} — IQR Outlier View"
    )

    ax.set_ylabel(feature)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "14_iqr_outlier_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. CLASS-WISE NUMERICAL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("CLASS-WISE NUMERICAL COMPARISON")
print("=" * 80)

comparison_features = [
    "BMI",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "Age",
    "Income",
]

class_comparison = (
    df.groupby("Diabetes_binary")[comparison_features]
    .agg(["mean", "median"])
    .round(3)
)

print(class_comparison)

class_comparison.to_csv(
    REPORT_DIR / "deep_eda_class_comparison.csv"
)


# ============================================================
# 10. CLASS-WISE BOXPLOTS
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(16, 10)
)

for ax, feature in zip(
    axes.flatten(),
    comparison_features
):

    sns.boxplot(
        data=df,
        x="Diabetes_binary",
        y=feature,
        ax=ax
    )

    ax.set_title(
        f"{feature} by Diabetes Status"
    )

    ax.set_xlabel(
        "Diabetes Status (0 = No, 1 = Yes)"
    )

    ax.set_ylabel(feature)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "15_classwise_feature_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. PREVALENCE BY IMPORTANT ORDINAL VARIABLES
# ============================================================

print("\n" + "=" * 80)
print("DIABETES PREVALENCE BY SELECTED FEATURES")
print("=" * 80)

prevalence_features = [
    "GenHlth",
    "Age",
    "Education",
    "Income",
]

for feature in prevalence_features:

    prevalence = (
        df.groupby(feature)["Diabetes_binary"]
        .mean()
        .mul(100)
        .reset_index()
    )

    prevalence.columns = [
        feature,
        "Diabetes_Prevalence_Percent",
    ]

    print(f"\n--- {feature} ---")
    print(
        prevalence.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    safe_name = feature.lower()

    prevalence.to_csv(
        REPORT_DIR
        / f"deep_eda_prevalence_{safe_name}.csv",
        index=False
    )


# ============================================================
# 12. PREVALENCE VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(14, 10)
)

for ax, feature in zip(
    axes.flatten(),
    prevalence_features
):

    prevalence = (
        df.groupby(feature)["Diabetes_binary"]
        .mean()
        .mul(100)
    )

    ax.bar(
        prevalence.index.astype(str),
        prevalence.values
    )

    ax.set_title(
        f"Diabetes Prevalence by {feature}"
    )

    ax.set_xlabel(feature)

    ax.set_ylabel(
        "Diabetes Prevalence (%)"
    )

    ax.tick_params(axis="x", rotation=45)

    # Add percentage labels
    for i, value in enumerate(prevalence.values):

        ax.text(
            i,
            value + 0.5,
            f"{value:.1f}%",
            ha="center",
            fontsize=8
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "16_diabetes_prevalence_by_features.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 13. IMPORTANT BINARY FEATURE COMPARISON
# ============================================================

binary_features = [
    "HighBP",
    "HighChol",
    "HeartDiseaseorAttack",
    "Stroke",
    "DiffWalk",
    "PhysActivity",
]

binary_records = []

for feature in binary_features:

    prevalence = (
        df.groupby(feature)["Diabetes_binary"]
        .mean()
        .mul(100)
    )

    for category, percentage in prevalence.items():

        binary_records.append(
            {
                "Feature": feature,
                "Category": int(category),
                "Diabetes_Prevalence_Percent": percentage,
            }
        )

binary_prevalence = pd.DataFrame(binary_records)

print("\nBinary feature prevalence:")
print(
    binary_prevalence.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

binary_prevalence.to_csv(
    REPORT_DIR / "deep_eda_binary_feature_prevalence.csv",
    index=False
)


# ============================================================
# 14. CORRELATION RANKING
# ============================================================

print("\n" + "=" * 80)
print("TARGET CORRELATION RANKING")
print("=" * 80)

correlations = (
    df.corr(numeric_only=True)["Diabetes_binary"]
    .drop("Diabetes_binary")
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
)

correlation_table = pd.DataFrame(
    {
        "Feature": correlations.index,
        "Correlation": correlations.values,
        "Absolute_Correlation": correlations.abs().values,
    }
)

print(
    correlation_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

correlation_table.to_csv(
    REPORT_DIR / "deep_eda_correlation_ranking.csv",
    index=False
)


# ============================================================
# 15. FEATURE CARDINALITY SUMMARY
# ============================================================

binary_like_features = []

for feature in df.columns:

    unique_count = df[feature].nunique()

    if unique_count <= 2:

        binary_like_features.append(feature)

print("\n" + "=" * 80)
print("BINARY-LIKE FEATURES")
print("=" * 80)

print(
    f"Number of binary-like features: "
    f"{len(binary_like_features)}"
)

print(binary_like_features)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("DEEP EDA COMPLETED")
print("=" * 80)

print(f"Original rows: {len(df):,}")
print(
    f"Rows after duplicate removal: "
    f"{len(df_no_duplicates):,}"
)

print(f"Duplicates removed: {duplicate_total:,}")
print(f"Missing values: {missing_total:,}")

print("\nGenerated visualizations:")

generated_files = [
    "14_iqr_outlier_analysis.png",
    "15_classwise_feature_comparison.png",
    "16_diabetes_prevalence_by_features.png",
]

for filename in generated_files:

    path = OUTPUT_DIR / filename

    print(
        f"  {'✓' if path.exists() else '✗'} "
        f"{filename}"
    )

print("\nGenerated analysis files:")

analysis_files = [
    "deep_eda_target_distribution.csv",
    "deep_eda_duplicate_impact.csv",
    "deep_eda_numeric_summary.csv",
    "deep_eda_unique_values.csv",
    "deep_eda_outlier_analysis.csv",
    "deep_eda_class_comparison.csv",
    "deep_eda_binary_feature_prevalence.csv",
    "deep_eda_correlation_ranking.csv",
]

for filename in analysis_files:

    path = REPORT_DIR / filename

    print(
        f"  {'✓' if path.exists() else '✗'} "
        f"{filename}"
    )

print("=" * 80)