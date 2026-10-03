"""
Week 5 - Deep Learning Application in Data Science

Step 5: Evaluate the trained neural network on the test dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_mlp_best.pt"

MODELS_DIR = PROJECT_ROOT / "models"
VIS_DIR = PROJECT_ROOT / "visualizations"

MODELS_DIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_SEED = 42
BATCH_SIZE = 1024
TARGET_COLUMN = "Diabetes_binary"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. MODEL DEFINITION
# ============================================================

class DiabetesMLP(nn.Module):

    def __init__(self, input_features):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):

        return self.network(x).squeeze(1)


# ============================================================
# 4. LOAD TEST DATA
# ============================================================

print("=" * 70)
print("MODEL EVALUATION ON TEST DATA")
print("=" * 70)

print(f"Device: {DEVICE}")

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_PATH)

X_test = test_df.drop(
    columns=[TARGET_COLUMN]
).values.astype(np.float32)

y_test = test_df[TARGET_COLUMN].values.astype(np.float32)

print(f"Test features shape: {X_test.shape}")
print(f"Test target shape  : {y_test.shape}")


# ============================================================
# 5. CONVERT TO TENSORS
# ============================================================

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
)


# ============================================================
# 6. CREATE TEST DATALOADER
# ============================================================

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 7. LOAD TRAINED MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

input_features = checkpoint["input_features"]

model = DiabetesMLP(
    input_features=input_features
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nBest model loaded successfully.")

print(f"Best training epoch: {checkpoint['best_epoch']}")


# ============================================================
# 8. GENERATE TEST PREDICTIONS
# ============================================================

all_targets = []
all_probabilities = []

print("\nGenerating predictions...")

with torch.no_grad():

    for batch_X, batch_y in test_loader:

        batch_X = batch_X.to(DEVICE)

        logits = model(batch_X)

        probabilities = torch.sigmoid(logits)

        all_targets.extend(
            batch_y.numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


y_true = np.array(all_targets).astype(int)

y_probability = np.array(
    all_probabilities
)

# Default classification threshold
THRESHOLD = 0.50

y_pred = (
    y_probability >= THRESHOLD
).astype(int)


# ============================================================
# 9. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

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

roc_auc = roc_auc_score(
    y_true,
    y_probability
)


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEST SET RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "No Diabetes",
        "Diabetes"
    ],
    zero_division=0
)

print(report)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

tn, fp, fn, tp = cm.ravel()

print(f"\nTrue Negatives  : {tn}")
print(f"False Positives : {fp}")
print(f"False Negatives : {fn}")
print(f"True Positives  : {tp}")


# ============================================================
# 13. SAVE TEST RESULTS
# ============================================================

results = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "threshold": THRESHOLD,
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp)
}

results_df = pd.DataFrame(
    [results]
)

results_path = MODELS_DIR / "test_results.csv"

results_df.to_csv(
    results_path,
    index=False
)

print(
    f"\nTest results saved to: {results_path}"
)


# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "actual": y_true,
    "probability": y_probability,
    "prediction": y_pred
})

predictions_path = MODELS_DIR / "test_predictions.csv"

predictions_df.to_csv(
    predictions_path,
    index=False
)

print(
    f"Test predictions saved to: {predictions_path}"
)


# ============================================================
# 15. CONFUSION MATRIX VISUALIZATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Diabetes",
        "Diabetes"
    ]
).plot(
    ax=ax
)

ax.set_title(
    "Confusion Matrix - Diabetes Prediction"
)

plt.tight_layout()

cm_path = VIS_DIR / "09_confusion_matrix.png"

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Confusion matrix saved to: {cm_path}"
)


# ============================================================
# 16. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Neural Network (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Diabetes Prediction"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

roc_path = VIS_DIR / "10_roc_curve.png"

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"ROC curve saved to: {roc_path}"
)


# ============================================================
# 17. SAVE PREDICTION DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    y_probability[y_true == 0],
    bins=50,
    alpha=0.6,
    label="No Diabetes"
)

plt.hist(
    y_probability[y_true == 1],
    bins=50,
    alpha=0.6,
    label="Diabetes"
)

plt.axvline(
    THRESHOLD,
    linestyle="--",
    label="Threshold = 0.50"
)

plt.xlabel(
    "Predicted Probability"
)

plt.ylabel(
    "Number of Samples"
)

plt.title(
    "Predicted Probability Distribution"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

probability_path = (
    VIS_DIR /
    "11_prediction_probability_distribution.png"
)

plt.savefig(
    probability_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Probability distribution saved to: "
    f"{probability_path}"
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 COMPLETED")
print("=" * 70)

print("Test evaluation completed successfully.")

print("\nFinal test metrics:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion matrix:")
print(cm)

print("\nGenerated files:")
print("- models/test_results.csv")
print("- models/test_predictions.csv")
print("- visualizations/09_confusion_matrix.png")
print("- visualizations/10_roc_curve.png")
print("- visualizations/11_prediction_probability_distribution.png")