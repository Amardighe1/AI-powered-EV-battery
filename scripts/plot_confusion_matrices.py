"""Plot confusion matrix heatmaps for all Phase 2 models."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = ["Normal (0)", "Abnormal (1)", "Elec Fault (2)", "Therm Fault (3)", "Early Warn (4)", "Critical (5)"]

def plot_cm(cm, title, out_path):
    cm = np.array(cm)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall / Row-Normalized Proportion")
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label',
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Annotate cells with raw count and percentage
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_str = f"{val:,}\n({pct:.1f}%)" if val > 0 else "0"
            ax.text(
                j, i, text_str,
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=8,
            )
            
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def main():
    with open("artifacts/metadata.json", "r") as f:
        meta = json.load(f)
        
    os.makedirs("plots", exist_ok=True)
    
    # 1. Threshold
    plot_cm(
        meta["models"]["threshold"]["test_metrics"]["confusion_matrix"],
        "Test Confusion Matrix: Threshold Baseline",
        "plots/confusion_matrix_threshold.png"
    )
    # 2. Random Forest
    plot_cm(
        meta["models"]["random_forest"]["test_metrics"]["confusion_matrix"],
        "Test Confusion Matrix: Balanced Random Forest (Best Model)",
        "plots/confusion_matrix_random_forest.png"
    )
    # 3. 1D-CNN
    plot_cm(
        meta["models"]["1d_cnn"]["test_metrics"]["confusion_matrix"],
        "Test Confusion Matrix: 1D-CNN (CPU)",
        "plots/confusion_matrix_1d_cnn.png"
    )
    # 4. Ablation
    plot_cm(
        meta["models"]["ablation_no_gas"]["test_metrics"]["confusion_matrix"],
        "Test Confusion Matrix: Random Forest (No Gas Sensors)",
        "plots/confusion_matrix_ablation_no_gas.png"
    )

if __name__ == "__main__":
    main()

