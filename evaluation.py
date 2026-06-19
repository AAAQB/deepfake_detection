import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader

# Import global configuration from pipeline
from deepfake_pipeline import C

# =====================================================================
# Configuration Alignment
# =====================================================================
TARGET_DATASET_DIR = "data_face"
C.experiment_name = "deepfake_v1"


# =====================================================================


class NpyEvaluationDataset(Dataset):
    """
    Custom evaluation dataset aligned with deepfake_pipeline output structure.
    Scans data_face/{image,video}/{real,filter,deepfake} for .npy feature matrices.
    """

    def __init__(self, root_dir, classes):
        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.samples = []

        if not os.path.exists(root_dir):
            return

        for modality in ["image", "video"]:
            modality_dir = os.path.join(root_dir, modality)
            if not os.path.isdir(modality_dir):
                continue
            for cls in self.classes:
                cls_dir = os.path.join(modality_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                label = self.class_to_idx[cls]

                for root, _, files in os.walk(cls_dir):
                    for f in files:
                        if f.endswith(".npy"):
                            self.samples.append((os.path.join(root, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = torch.from_numpy(np.load(path)).float()
        return x, torch.tensor(label, dtype=torch.long)


def evaluate_model(mode_type="frame"):
    """
    Evaluates the trained model on specified mode and generates metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode_type != "frame":
        print(f"Error: Mode '{mode_type}' is not supported.")
        return

    if not os.path.exists(TARGET_DATASET_DIR):
        print(f"Error: Dataset directory '{TARGET_DATASET_DIR}' not found.")
        return

    test_dataset = NpyEvaluationDataset(root_dir=TARGET_DATASET_DIR, classes=C.classes)

    if len(test_dataset) == 0:
        print(f"Error: No .npy files found in '{TARGET_DATASET_DIR}'.")
        return

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Load Model Architecture and Weights
    from deepfake_pipeline import EfficientNetClassifier
    model = EfficientNetClassifier(num_classes=C.num_classes, dropout=C.dropout).to(device)

    ckpt_p = os.path.join(C.checkpoint_dir, f"frame_{C.experiment_name}", "best_model.pt")
    if not os.path.exists(ckpt_p):
        print(f"Error: Model weights checkpoint not found at '{ckpt_p}'.")
        return

    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    # Inference Loop
    with torch.no_grad():
        for batch in test_loader:
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

    # Output strict evaluation matrix
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes, digits=4))

    # Export high-resolution visualization charts
    generate_charts(y_true, y_pred, y_prob, mode_type, test_dataset.classes)


def generate_charts(y_true, y_pred, y_prob, mode_type, class_names):
    """Generates and saves professional evaluation charts with English labels."""
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {mode_type.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = f"evaluation_results_{mode_type}_cm.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. ROC & AUC Curve
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    plt.figure(figsize=(7, 6))
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

    # Minimalist output paths info
    print(f"Confusion Matrix saved to: {cm_path}")
    print(f"ROC Curve saved to: {roc_path}")


if __name__ == "__main__":
    evaluate_model(mode_type="frame")
