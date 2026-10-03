# ================================================================
# STEP 6: THRESHOLD ANALYSIS
# Diabetes Health Indicators - Deep Learning Project
# ================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ================================================================
# CONFIGURATION
# ================================================================

PREDICTIONS_PATH = "models/test_predictions.csv"
RESULTS_PATH = "models/threshold_analysis.csv"
PLOT_PATH = "visualizations/12_threshold_analysis.png"

THRESHOLDS = np.arange(0.20, 0.81, 0.05)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)


# ================================================================
# 1. LOAD TEST PREDICTIONS
# ================================================================

print("\nLoading test predictions...")

if not os.path.exists(PREDICTIONS_PATH):
    raise FileNotFoundError(
        f"Prediction file not found: {PREDICTIONS_PATH}"
    )

df = pd.read_csv(PREDICTIONS_PATH)

print("Prediction file loaded successfully.")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ================================================================
# 2. IDENTIFY TARGET AND PROBABILITY COLUMNS
# ================================================================

# Expected columns from the evaluation script:
# actual, probability, prediction

required_columns = ["actual", "probability"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' not found in prediction file."
        )

y_true = df["actual"].astype(int).values
y_probability = df["probability"].astype(float).values


print("\nTarget distribution:")
print(pd.Series(y_true).value_counts().sort_index())

print("\nProbability range:")
print(f"Minimum: {y_probability.min():.6f}")
print(f"Maximum: {y_probability.max():.6f}")


# ================================================================
# 3. CALCULATE METRICS FOR EACH THRESHOLD
# ================================================================

print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

results = []

for threshold in THRESHOLDS:

    y_pred = (y_probability >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": round(float(threshold), 2),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp)
    })


results_df = pd.DataFrame(results)


# ================================================================
# 4. DISPLAY RESULTS
# ================================================================

display_df = results_df.copy()

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]:
    display_df[column] = display_df[column].map(
        lambda x: f"{x:.4f}"
    )

print("\n")
print(display_df.to_string(index=False))


# ================================================================
# 5. IDENTIFY MAXIMUM F1 THRESHOLD
# ================================================================

best_f1_index = results_df["F1_Score"].idxmax()

best_f1_row = results_df.loc[best_f1_index]

print("\n" + "=" * 70)
print("HIGHEST F1 SCORE IN TEST THRESHOLD SWEEP")
print("=" * 70)

print(f"Threshold : {best_f1_row['Threshold']:.2f}")
print(f"Accuracy  : {best_f1_row['Accuracy']:.4f}")
print(f"Precision : {best_f1_row['Precision']:.4f}")
print(f"Recall    : {best_f1_row['Recall']:.4f}")
print(f"F1 Score  : {best_f1_row['F1_Score']:.4f}")


# ================================================================
# 6. SHOW 0.50 BASELINE
# ================================================================

baseline = results_df[
    np.isclose(results_df["Threshold"], 0.50)
]

if not baseline.empty:

    baseline = baseline.iloc[0]

    print("\n" + "=" * 70)
    print("CURRENT 0.50 THRESHOLD")
    print("=" * 70)

    print(f"Accuracy  : {baseline['Accuracy']:.4f}")
    print(f"Precision : {baseline['Precision']:.4f}")
    print(f"Recall    : {baseline['Recall']:.4f}")
    print(f"F1 Score  : {baseline['F1_Score']:.4f}")


# ================================================================
# 7. SAVE RESULTS
# ================================================================

os.makedirs("models", exist_ok=True)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print(f"\nThreshold results saved to:")
print(os.path.abspath(RESULTS_PATH))


# ================================================================
# 8. CREATE THRESHOLD PERFORMANCE PLOT
# ================================================================

os.makedirs("visualizations", exist_ok=True)

plt.figure(figsize=(10, 6))

plt.plot(
    results_df["Threshold"],
    results_df["Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["F1_Score"],
    marker="o",
    label="F1 Score"
)

plt.axvline(
    x=0.50,
    linestyle="--",
    label="Current Threshold = 0.50"
)

plt.xlabel("Prediction Threshold")
plt.ylabel("Metric Value")
plt.title("Threshold Analysis - Diabetes Classification")
plt.xticks(THRESHOLDS)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Threshold analysis plot saved to:")
print(os.path.abspath(PLOT_PATH))


# ================================================================
# 9. SAVE SUMMARY JSON
# ================================================================

summary = {
    "analysis": "Prediction threshold analysis",
    "thresholds_tested": [
        round(float(x), 2) for x in THRESHOLDS
    ],
    "baseline_threshold": 0.50,
    "highest_f1_threshold": float(
        best_f1_row["Threshold"]
    ),
    "highest_f1_score": float(
        best_f1_row["F1_Score"]
    )
}

with open(
    "models/threshold_analysis_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )

print(
    "Threshold summary saved to: "
    "models/threshold_analysis_summary.json"
)


# ================================================================
# COMPLETION
# ================================================================

print("\n" + "=" * 70)
print("STEP 6 COMPLETED")
print("=" * 70)

print("\nGenerated files:")
print("- models/threshold_analysis.csv")
print("- models/threshold_analysis_summary.json")
print("- visualizations/12_threshold_analysis.png")