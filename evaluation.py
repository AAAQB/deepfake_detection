import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader

# Import baseline pipeline configurations and data scanner modules
from deepfake_pipeline import (
    C,
    EfficientNetClassifier,
    TemporalModel,
    TemporalDataset,
    _scan_face_root
)

# =====================================================================
# Pipeline Global Settings Alignment
# =====================================================================
TARGET_DATASET_DIR = "data_face"
C.experiment_name = "deepfake_v1"
C.face_root = TARGET_DATASET_DIR


class EvaluationImageDataset(Dataset):
    """Isolated evaluation dataset mapping strictly onto independent image matrices."""

    def __init__(self, image_items):
        self.image_items = image_items

    def __len__(self):
        return len(self.image_items)

    def __getitem__(self, idx):
        path, label = self.image_items[idx]
        x = torch.from_numpy(np.load(path)).float()
        return x, torch.tensor(label, dtype=torch.long)


def generate_loss_chart(mode_type):
    """Parses TensorBoard event logs to generate clean training vs validation loss curves."""
    log_dir = os.path.join("logs", f"{mode_type}_{C.experiment_name}")
    if not os.path.exists(log_dir):
        print(f"[Info] TensorBoard log directory not found for {mode_type}. Skipping loss curve.")
        return

    try:
        from tensorboard.backend.event_processing import event_accumulator
        event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "events.out.tfevents" in f]
        if not event_files:
            print(f"[Info] No event logs discovered for {mode_type}. Skipping loss curve.")
            return

        # Extract scalar history matrices from latest tracking log
        ea = event_accumulator.EventAccumulator(event_files[-1])
        ea.Reload()

        if "train/loss" not in ea.Scalars() or "val/loss" not in ea.Scalars():
            print(f"[Info] Loss tracking parameters incomplete for {mode_type}. Skipping loss curve.")
            return

        train_vals = [s.value for s in ea.Scalars("train/loss")]
        val_vals = [s.value for s in ea.Scalars("val/loss")]

        num_epochs = len(val_vals)
        if num_epochs > 0 and len(train_vals) > 0:
            # Aggregate step-level training loss values into clean epoch buckets
            chunks = np.array_split(train_vals, num_epochs)
            epoch_train_vals = [np.mean(c) for c in chunks if len(c) > 0]
            epochs = list(range(1, len(epoch_train_vals) + 1))

            plt.figure(figsize=(6, 4.5))
            plt.plot(epochs, epoch_train_vals, label="Train Loss", color="royalblue", marker='o', linewidth=2)
            plt.plot(range(1, num_epochs + 1), val_vals, label="Val Loss", color="darkorange", marker='s', linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Loss Convergence Curve - {mode_type.upper()}")
            plt.legend(loc="upper right")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            loss_path = f"evaluation_results_{mode_type}_loss.png"
            plt.savefig(loss_path, dpi=300)
            plt.close()
            print(f"Loss Curve saved to: {loss_path}")
    except Exception as e:
        print(f"[Warning] Could not generate loss chart for {mode_type}: {e}")


def generate_charts(y_true, y_pred, y_prob, mode_type, class_names):
    """Generates standard clean evaluation charts for academic reporting."""
    # 1. Confusion Matrix Plotting
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {mode_type.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = f"evaluation_results_{mode_type}_cm.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. ROC & AUC Vector Generation
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    plt.figure(figsize=(6, 5))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i].upper()} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='silver', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {mode_type.upper()}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = f"evaluation_results_{mode_type}_roc.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"Confusion Matrix saved to: {cm_path}")
    print(f"ROC Curve saved to: {roc_path}")


def evaluate_model(mode_type="frame"):
    """Runs evaluation pass using isolated test arrays for the selected architecture."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_items, video_dirs = _scan_face_root()

    # -----------------------------------------------------------------
    # Configuration Setup: Frame Mode (Images Only)
    # -----------------------------------------------------------------
    if mode_type == "frame":
        if len(image_items) == 0:
            print(f"[Error] No evaluation images found in '{TARGET_DATASET_DIR}/image'.")
            return
        test_dataset = EvaluationImageDataset(image_items)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
        model = EfficientNetClassifier(num_classes=C.num_classes, dropout=C.dropout).to(device)
        ckpt_p = os.path.join(C.checkpoint_dir, f"frame_{C.experiment_name}", "best_model.pt")

    # -----------------------------------------------------------------
    # Configuration Setup: Temporal Mode (Video Sequences Only)
    # -----------------------------------------------------------------
    elif mode_type == "temporal":
        if len(video_dirs) == 0:
            print(f"[Error] No evaluation video structures found in '{TARGET_DATASET_DIR}/video'.")
            return
        test_dataset = TemporalDataset(video_dirs, seq_len=C.temporal_seq_len, stride=C.temporal_stride)
        test_loader = DataLoader(test_dataset, batch_size=C.temporal_batch_size, shuffle=False, num_workers=0)
        model = TemporalModel(num_classes=C.num_classes, hidden_size=C.hidden_size,
                              num_layers=C.num_lstm_layers, dropout=C.lstm_dropout,
                              bidirectional=C.bidirectional).to(device)
        ckpt_p = os.path.join(C.checkpoint_dir, f"temporal_{C.experiment_name}", "best_model.pt")
    else:
        return

    # Checkpoint Integrity Check
    if not os.path.exists(ckpt_p):
        print(f"[Error] Missing checkpoint weights: {ckpt_p}")
        return

    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    # Concise Inference Execution Loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating [{mode_type.upper()}]", leave=False):
            x, y = batch
            x = x.to(device, non_blocking=True)

            outputs = model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_labels.extend(y.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Output Standard Metrics and Generate Graphics
    print(f"\n--- {mode_type.upper()} EVALUATION REPORT ---")
    print(classification_report(y_true, y_pred, target_names=C.classes, digits=4))
    generate_charts(y_true, y_pred, y_prob, mode_type, C.classes)
    generate_loss_chart(mode_type)


if __name__ == "__main__":
    # Execute batch test evaluation routines across both isolated sets sequentially
    evaluate_model(mode_type="frame")
    evaluate_model(mode_type="temporal")
