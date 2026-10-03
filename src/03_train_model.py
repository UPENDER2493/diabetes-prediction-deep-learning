"""
Week 5 - Deep Learning Application in Data Science

Step 4: Train a Neural Network for Diabetes Prediction

Dataset:
CDC Diabetes Health Indicators Dataset

Task:
Binary classification of diabetes status.

Framework:
PyTorch
"""

from pathlib import Path
import copy
import json

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
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"

MODELS_DIR = PROJECT_ROOT / "models"
VIS_DIR = PROJECT_ROOT / "visualizations"

MODELS_DIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("DEEP LEARNING MODEL TRAINING")
print("=" * 70)

print(f"PyTorch version : {torch.__version__}")
print(f"Training device : {DEVICE}")
print(f"Random seed     : {RANDOM_SEED}")


# ============================================================
# 4. LOAD PROCESSED DATA
# ============================================================

print("\nLoading processed datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

TARGET_COLUMN = "Diabetes_binary"

X_train = train_df.drop(columns=[TARGET_COLUMN]).values.astype(np.float32)
y_train = train_df[TARGET_COLUMN].values.astype(np.float32)

X_val = val_df.drop(columns=[TARGET_COLUMN]).values.astype(np.float32)
y_val = val_df[TARGET_COLUMN].values.astype(np.float32)

print(f"Training features shape   : {X_train.shape}")
print(f"Training target shape     : {y_train.shape}")
print(f"Validation features shape : {X_val.shape}")
print(f"Validation target shape   : {y_val.shape}")


# ============================================================
# 5. CONVERT DATA TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


# ============================================================
# 6. CREATE DATA LOADERS
# ============================================================

BATCH_SIZE = 1024

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"\nBatch size      : {BATCH_SIZE}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


# ============================================================
# 7. CALCULATE CLASS WEIGHT
# ============================================================

negative_count = np.sum(y_train == 0)
positive_count = np.sum(y_train == 1)

positive_weight = negative_count / positive_count

print("\nClass distribution:")
print(f"Class 0 samples : {negative_count}")
print(f"Class 1 samples : {positive_count}")
print(f"Positive weight : {positive_weight:.4f}")


# ============================================================
# 8. DEFINE NEURAL NETWORK
# ============================================================

class DiabetesMLP(nn.Module):
    """
    Multi-Layer Perceptron for binary diabetes classification.

    Architecture:
        Input(21)
        -> Linear(64)
        -> BatchNorm
        -> ReLU
        -> Dropout(0.30)
        -> Linear(32)
        -> BatchNorm
        -> ReLU
        -> Dropout(0.20)
        -> Linear(16)
        -> ReLU
        -> Linear(1)
    """

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


input_features = X_train.shape[1]

model = DiabetesMLP(input_features=input_features).to(DEVICE)

print("\nModel architecture:")
print(model)


# ============================================================
# 9. LOSS FUNCTION
# ============================================================

pos_weight_tensor = torch.tensor(
    [positive_weight],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight_tensor
)


# ============================================================
# 10. OPTIMIZER
# ============================================================

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Reduce learning rate when validation loss stops improving.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-6
)


# ============================================================
# 11. TRAINING SETTINGS
# ============================================================

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

best_model_state = copy.deepcopy(model.state_dict())

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_precision": [],
    "val_recall": [],
    "val_f1": [],
    "val_roc_auc": [],
    "learning_rate": []
}


# ============================================================
# 12. TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(1, MAX_EPOCHS + 1):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0
    train_samples = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(batch_X)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()

        batch_size_current = batch_X.size(0)

        running_train_loss += loss.item() * batch_size_current
        train_samples += batch_size_current

    train_loss = running_train_loss / train_samples

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    running_val_loss = 0.0
    val_samples = 0

    all_val_targets = []
    all_val_probabilities = []

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = model(batch_X)

            loss = criterion(logits, batch_y)

            probabilities = torch.sigmoid(logits)

            batch_size_current = batch_X.size(0)

            running_val_loss += loss.item() * batch_size_current
            val_samples += batch_size_current

            all_val_targets.extend(
                batch_y.cpu().numpy()
            )

            all_val_probabilities.extend(
                probabilities.cpu().numpy()
            )

    val_loss = running_val_loss / val_samples

    all_val_targets = np.array(all_val_targets)
    all_val_probabilities = np.array(all_val_probabilities)

    # Convert probabilities to binary predictions.
    predictions = (
        all_val_probabilities >= 0.50
    ).astype(int)

    val_accuracy = accuracy_score(
        all_val_targets,
        predictions
    )

    val_precision = precision_score(
        all_val_targets,
        predictions,
        zero_division=0
    )

    val_recall = recall_score(
        all_val_targets,
        predictions,
        zero_division=0
    )

    val_f1 = f1_score(
        all_val_targets,
        predictions,
        zero_division=0
    )

    val_roc_auc = roc_auc_score(
        all_val_targets,
        all_val_probabilities
    )

    # Update learning rate.
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)
    history["val_precision"].append(val_precision)
    history["val_recall"].append(val_recall)
    history["val_f1"].append(val_f1)
    history["val_roc_auc"].append(val_roc_auc)
    history["learning_rate"].append(current_lr)

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Accuracy: {val_accuracy:.4f} | "
            f"Precision: {val_precision:.4f} | "
            f"Recall: {val_recall:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"ROC-AUC: {val_roc_auc:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"BEST"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Accuracy: {val_accuracy:.4f} | "
            f"Precision: {val_precision:.4f} | "
            f"Recall: {val_recall:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"ROC-AUC: {val_roc_auc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print(
            f"\nEarly stopping triggered after "
            f"{EARLY_STOPPING_PATIENCE} epochs without "
            f"validation-loss improvement."
        )

        break


# ============================================================
# 13. RESTORE BEST MODEL
# ============================================================

model.load_state_dict(best_model_state)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"Best epoch     : {best_epoch}")
print(f"Best val loss  : {best_val_loss:.4f}")


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

model_path = MODELS_DIR / "diabetes_mlp_best.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_features": input_features,
        "architecture": {
            "input": input_features,
            "hidden_1": 64,
            "hidden_2": 32,
            "hidden_3": 16,
            "dropout_1": 0.30,
            "dropout_2": 0.20
        },
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "positive_weight": positive_weight,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    },
    model_path
)

print(f"Best model saved to: {model_path}")


# ============================================================
# 15. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_path = MODELS_DIR / "training_history.csv"

history_df.to_csv(
    history_path,
    index=False
)

print(f"Training history saved to: {history_path}")


# ============================================================
# 16. SAVE TRAINING CONFIGURATION
# ============================================================

config = {
    "random_seed": RANDOM_SEED,
    "device": str(DEVICE),
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "positive_class_weight": float(positive_weight),
    "input_features": input_features,
    "best_epoch": best_epoch,
    "best_validation_loss": float(best_val_loss)
}

config_path = MODELS_DIR / "training_config.json"

with open(config_path, "w") as file:
    json.dump(config, file, indent=4)

print(f"Training configuration saved to: {config_path}")


# ============================================================
# 17. PLOT TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    label="Validation Loss"
)

plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best Epoch ({best_epoch})"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

loss_plot_path = VIS_DIR / "07_training_validation_loss.png"

plt.savefig(
    loss_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Loss curve saved to: {loss_plot_path}")


# ============================================================
# 18. PLOT VALIDATION METRICS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["val_accuracy"],
    label="Accuracy"
)

plt.plot(
    history["epoch"],
    history["val_precision"],
    label="Precision"
)

plt.plot(
    history["epoch"],
    history["val_recall"],
    label="Recall"
)

plt.plot(
    history["epoch"],
    history["val_f1"],
    label="F1 Score"
)

plt.plot(
    history["epoch"],
    history["val_roc_auc"],
    label="ROC-AUC"
)

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Metrics During Training")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

metrics_plot_path = VIS_DIR / "08_validation_metrics.png"

plt.savefig(
    metrics_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Metrics curve saved to: {metrics_plot_path}")


# ============================================================
# 19. FINAL VALIDATION RESULTS
# ============================================================

best_row = history_df[
    history_df["epoch"] == best_epoch
].iloc[0]

print("\nBest validation results:")
print(f"Accuracy  : {best_row['val_accuracy']:.4f}")
print(f"Precision : {best_row['val_precision']:.4f}")
print(f"Recall    : {best_row['val_recall']:.4f}")
print(f"F1 Score  : {best_row['val_f1']:.4f}")
print(f"ROC-AUC   : {best_row['val_roc_auc']:.4f}")

print("\n" + "=" * 70)
print("STEP 4 COMPLETED")
print("=" * 70)