from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================
# PROFESSIONAL NEURAL NETWORK ARCHITECTURE DIAGRAM
# ============================================================

OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

output_path = OUTPUT_DIR / "13_network_architecture.png"


# ------------------------------------------------------------
# Architecture definition
# ------------------------------------------------------------

layers = [
    ("Input Layer", "21 Features", "Raw/standardized\nhealth indicators"),
    ("Linear Layer", "21 → 64", "64 neurons"),
    ("BatchNorm", "64", "Normalization"),
    ("ReLU", "64", "Activation"),
    ("Dropout", "30%", "Regularization"),
    ("Linear Layer", "64 → 32", "32 neurons"),
    ("BatchNorm", "32", "Normalization"),
    ("ReLU", "32", "Activation"),
    ("Dropout", "20%", "Regularization"),
    ("Linear Layer", "32 → 16", "16 neurons"),
    ("ReLU", "16", "Activation"),
    ("Output Layer", "16 → 1", "Diabetes\nprobability"),
]


# ------------------------------------------------------------
# Figure setup
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(15, 18))

ax.set_xlim(0, 10)
ax.set_ylim(0, len(layers) * 1.35 + 1)
ax.axis("off")


# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

ax.text(
    5,
    len(layers) * 1.35 + 0.45,
    "Diabetes Prediction Neural Network Architecture",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
)

ax.text(
    5,
    len(layers) * 1.35 - 0.15,
    "PyTorch Multilayer Perceptron (MLP)",
    ha="center",
    va="center",
    fontsize=13,
)


# ------------------------------------------------------------
# Draw layers
# ------------------------------------------------------------

box_width = 5.8
box_height = 0.85
x_position = 2.1

y_positions = []

start_y = len(layers) * 1.35 - 1.25

for index, (layer_name, layer_size, description) in enumerate(layers):

    y = start_y - index * 1.35
    y_positions.append(y)

    box = FancyBboxPatch(
        (x_position, y),
        box_width,
        box_height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.5,
        edgecolor="black",
        facecolor="white",
    )

    ax.add_patch(box)

    # Layer name
    ax.text(
        x_position + 0.25,
        y + 0.57,
        layer_name,
        fontsize=11,
        fontweight="bold",
        va="center",
    )

    # Layer dimensions
    ax.text(
        x_position + box_width - 0.25,
        y + 0.57,
        layer_size,
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="center",
    )

    # Description
    ax.text(
        x_position + 0.25,
        y + 0.23,
        description,
        fontsize=9,
        va="center",
    )


# ------------------------------------------------------------
# Draw arrows between layers
# ------------------------------------------------------------

for i in range(len(layers) - 1):

    current_y = y_positions[i]
    next_y = y_positions[i + 1]

    arrow = FancyArrowPatch(
        (5, current_y),
        (5, next_y + box_height),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.4,
    )

    ax.add_patch(arrow)


# ------------------------------------------------------------
# Side annotations
# ------------------------------------------------------------

annotation_x = 8.25

annotations = [
    (0, "21 input variables"),
    (1, "Feature transformation"),
    (2, "Stabilizes activations"),
    (3, "Non-linear learning"),
    (4, "Reduces overfitting"),
    (5, "Learns compact representation"),
    (6, "Stabilizes hidden layer"),
    (7, "Non-linear learning"),
    (8, "Additional regularization"),
    (9, "Higher-level representation"),
    (10, "Non-linear learning"),
    (11, "Binary classification output"),
]


for index, annotation in annotations:

    y = y_positions[index] + box_height / 2

    ax.text(
        annotation_x,
        y,
        annotation[1],
        fontsize=9,
        va="center",
        ha="left",
    )


# ------------------------------------------------------------
# Training configuration box
# ------------------------------------------------------------

config_y = 0.25

config_text = (
    "Training Configuration\n"
    "• Loss: Weighted BCEWithLogitsLoss\n"
    "• Optimizer: AdamW\n"
    "• Initial Learning Rate: 0.001\n"
    "• Weight Decay: 0.0001\n"
    "• Batch Size: 1024\n"
    "• Maximum Epochs: 30\n"
    "• Early Stopping Patience: 7\n"
    "• Random Seed: 42"
)

ax.text(
    5,
    config_y,
    config_text,
    ha="center",
    va="center",
    fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.6",
        linewidth=1.2,
        facecolor="white",
        edgecolor="black",
    ),
)


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("=" * 70)
print("NETWORK ARCHITECTURE DIAGRAM CREATED")
print("=" * 70)
print(f"Saved to: {output_path.resolve()}")
print("Resolution: 300 DPI")
print("=" * 70)