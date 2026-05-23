import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Smoothly import the configuration, models, and precise data loaders from the team's modules
from train import CFG, build_model, create_loaders
from interface import ensemble_predict


def evaluate_model(mode_type="frame"):
    """
    Evaluates specific models based on target mode.
    mode_type options: 'frame' | 'temporal' | 'ensemble'
    """
    device = torch.device(CFG.device)
    print(f"\n>>> Initialising Benchmark Suite for [{mode_type.upper()}] mode on {device}...")

    # 1. Dispatch correct data loader to preserve test integrity
    is_temporal = (mode_type == "temporal")
    # For ensemble testing, we fetch the frame dataset to test single-frame extraction capability
    loader_temporal_flag = True if mode_type == "temporal" else False
    _, _, test_loader = create_loaders(temporal=loader_temporal_flag)

    # 2. Reconstruct configurations and load model weights safely
    frame_model, temporal_model = None, None

    if mode_type in ["frame", "ensemble"]:
        frame_model = build_model(temporal=False).to(device)
        ckpt_p = os.path.join(CFG.checkpoint_dir, f"frame_{CFG.experiment_name}", "best_model.pt")
        if not os.path.exists(ckpt_p):
            print(f"Abort: Missing frame model weights at {ckpt_p}");
            return
        frame_model.load_state_dict(torch.load(ckpt_p, map_location=device))
        frame_model.eval()

    if mode_type in ["temporal", "ensemble"]:
        temporal_model = build_model(temporal=True).to(device)
        ckpt_p = os.path.join(CFG.checkpoint_dir, f"temporal_{CFG.experiment_name}", "best_model.pt")
        if not os.path.exists(ckpt_p):
            print(f"Abort: Missing temporal model weights at {ckpt_p}");
            return
        temporal_model.load_state_dict(torch.load(ckpt_p, map_location=device))
        temporal_model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    # 3. Execution loop across the target unseen data split
    print(f"  Processing evaluation rows via {mode_type} infrastructure...")
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch

            if mode_type == "frame":
                x = x.to(device, non_blocking=True)
                outputs = frame_model(x)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)

            elif mode_type == "temporal":
                x = x.to(device, non_blocking=True)
                outputs = temporal_model(x)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)

            elif mode_type == "ensemble":
                # Processes elementwise over the batch to leverage soft voting array matrices safely
                probs_list = []
                for sample in x:
                    # Input matrix conversion from tensor back to HWC numpy format expected by interface
                    sample_np = sample.numpy().transpose(1, 2, 0)
                    prob = ensemble_predict(frame_model, temporal_model, sample_np, device=str(device))
                    probs_list.append(prob)
                probs = np.array(probs_list)
                preds = probs.argmax(axis=1)

            all_labels.extend(y.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # 4. Standard evaluation outputs to console
    print(f"\n================= {mode_type.upper()} REPORT MATRIX =================")
    print(classification_report(y_true, y_pred, target_names=CFG.classes, digits=4))

    # 5. Export high-resolution chart configurations
    generate_charts(y_true, y_pred, y_prob, mode_type)
    print(f"==================================================================\n")


def generate_charts(y_true, y_pred, y_prob, mode_type):
    """Generates and saves professional analysis visualisations."""
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples' if mode_type == 'ensemble' else 'Blues',
                xticklabels=CFG.classes, yticklabels=CFG.classes)
    plt.title(f'Confusion Matrix — {mode_type.upper()} Performance')
    plt.ylabel('True Class Target')
    plt.xlabel('Predicted Classifier Selection')
    plt.tight_layout()
    cm_path = f"evaluation_results_{mode_type}_cm.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Plot Multi-Class ROC & compute AUC scores
    y_true_bin = label_binarize(y_true, classes=list(range(CFG.num_classes)))
    plt.figure(figsize=(7, 6))
    for i in range(CFG.num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{CFG.classes[i].upper()} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='silver', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC & AUC Metrics — {mode_type.upper()}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = f"evaluation_results_{mode_type}_roc.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"  [Metrics Exported] Charts successfully saved:\n    -> {cm_path}\n    -> {roc_path}")


if __name__ == "__main__":
    # Sequentially benchmark all deployed runtime variations
    evaluate_model(mode_type="frame")
    evaluate_model(mode_type="temporal")
    evaluate_model(mode_type="ensemble")