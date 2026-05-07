"""
- Confusion Matrix
- Precision, Recall, F1-Score (per class and weighted)
- ROC Curve (One-vs-Rest)
- Accuracy
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import EfficientNet as EN
import LSTM as LS
from dataset import FaceFrameDataset, CLASS_NAMES

sns.set()
COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]


@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    model.eval()
    all_logits = []
    all_labels = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(inputs)

        all_logits.append(outputs.cpu())
        all_labels.append(targets.cpu())

    logits = torch.cat(all_logits)
    probs = F.softmax(logits, dim=1).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_pred = np.argmax(probs, axis=1)

    return probs, y_pred, y_true


def compute_metrics(y_true, y_pred, probs):
    n_classes = len(CLASS_NAMES)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    # Weighted metrics
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    cls_prec, cls_rec, cls_f1, cls_support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    # Per-class AUC
    y_true_bin = np.eye(n_classes)[y_true]
    cls_auc = []
    for i in range(n_classes):
        try:
            cls_auc.append(roc_auc_score(y_true_bin[:, i], probs[:, i]))
        except Exception:
            cls_auc.append(0.0)

    return {
        "accuracy": float(acc),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(cls_prec[i]),
                "recall": float(cls_rec[i]),
                "f1": float(cls_f1[i]),
                "auc": float(cls_auc[i]),
                "support": int(cls_support[i]),
            }
            for i in range(n_classes)
        },
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                vmin=0, vmax=1, cbar_kws={"label": "Recall"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_true, probs, save_path):
    n_classes = len(CLASS_NAMES)
    y_true_bin = np.eye(n_classes)[y_true]

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        auc = roc_auc_score(y_true_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, lw=2, color=COLORS[i],
                label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="dataset_faces")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available() and not args.no_amp
    print(f"Device: {device}  |  AMP: {amp}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model_type = ckpt_args.get("model_type", "efficientnet")

    if model_type == "efficientnet":
        model = EN.EfficientNet(num_classes=3)
    else:
        model = LS.CNNLSTM(num_classes=3, seq_len=args.seq_len)

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type.upper()} ({params:,} params)")

    # Data
    seq_len = args.seq_len if model_type == "temporal" else None
    dataset = FaceFrameDataset(args.data_root, seq_len=seq_len, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Evaluate
    print(f"Evaluating on {len(dataset)} samples...")
    probs, y_pred, y_true = evaluate(model, loader, device, amp=amp)
    metrics = compute_metrics(y_true, y_pred, probs)

    # Print
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS (PDF Section 13)")
    print(f"{'='*50}")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision (w):   {metrics['precision_weighted']:.4f}")
    print(f"Recall (w):      {metrics['recall_weighted']:.4f}")
    print(f"F1 Score (w):    {metrics['f1_weighted']:.4f}\n")
    print(f"{'Class':>10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Support':>8}")
    print("-" * 52)
    for name in CLASS_NAMES:
        pc = metrics["per_class"][name]
        print(f"{name:>10} {pc['precision']:>8.4f} {pc['recall']:>8.4f} "
              f"{pc['f1']:>8.4f} {pc['auc']:>8.4f} {pc['support']:>8}")
    cm = np.array(metrics["confusion_matrix"])
    print(f"\nConfusion Matrix:")
    print(f"{'':>10}", "  ".join(f"{n:>8}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        print(f"{n:>10}", "  ".join(f"{cm[i,j]:>8}" for j in range(3)))

    # Plots
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.checkpoint))[0]
    plot_confusion_matrix(cm, os.path.join(args.output_dir, f"{base}_confusion.png"))
    plot_roc_curves(y_true, probs, os.path.join(args.output_dir, f"{base}_roc.png"))

    # Save JSON
    with open(os.path.join(args.output_dir, f"{base}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nPlots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
