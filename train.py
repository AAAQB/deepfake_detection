"""
train.py — Training pipeline for deepfake detection models.

Supports:
  - Frame-level (EfficientNet-B4) and temporal (EfficientNet-B4 + Bidirectional LSTM)
  - Automatic Mixed Precision (AMP)
  - Label smoothing
  - Cosine annealing with linear warmup
  - Gradient clipping
  - Early stopping
  - TensorBoard logging
  - Top-K checkpoint management
"""

import os
import sys
import time
import math
import random
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─── Existing project modules (unchanged) ──────────────────────────
from EfficientNet import EfficientNetClassifier
from LSTM import TemporalModel
from preprocessing import preprocess_train, preprocess_infer


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Central training hyperparameters."""
    # Paths
    dataset_faces: str = "dataset_faces"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    experiment_name: str = "deepfake_detection"

    # Classes
    classes: List[str] = field(default_factory=lambda: ["real", "filter", "deepfake"])
    num_classes: int = 3

    # Data
    batch_size: int = 32
    temporal_batch_size: int = 8
    num_workers: int = 4
    temporal_seq_len: int = 16
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42

    # Model
    backbone: str = "efficientnet_b4"
    pretrained: bool = True
    dropout: float = 0.4
    freeze_backbone: bool = False
    hidden_size: int = 512
    num_lstm_layers: int = 4
    lstm_dropout: float = 0.4
    bidirectional: bool = True

    # Optimisation
    optimizer: str = "adamw"          # adamw | adam | sgd
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9

    # Schedule
    scheduler: str = "cosine"         # cosine | plateau | step | none
    lr_min: float = 1e-6
    lr_patience: int = 5
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    warmup_epochs: int = 3

    # Regularisation
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 1.0
    max_epochs: int = 40
    early_stop_patience: int = 10
    amp: bool = True
    save_top_k: int = 3

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_dirs(self):
        for d in [self.checkpoint_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)


CFG = TrainingConfig()


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

class FaceFrameDataset(Dataset):
    """Loads pre-extracted .npy face tensors from dataset_faces/{class}/*/*.npy."""

    def __init__(self, root: str, class_names: List[str], mode: str = "train"):
        self.root = root
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.mode = mode
        self.samples: List[Tuple[str, int]] = []
        self._scan(class_names)
        if not self.samples:
            raise RuntimeError(f"No samples found under {root}")

    def _scan(self, class_names: List[str]):
        for cls_name in class_names:
            cls_dir = os.path.join(self.root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            label = self.class_to_idx[cls_name]
            for vid in sorted(os.listdir(cls_dir)):
                vid_dir = os.path.join(cls_dir, vid)
                if not os.path.isdir(vid_dir):
                    continue
                for fname in sorted(os.listdir(vid_dir)):
                    if fname.endswith((".npy", ".jpg", ".png", ".jpeg")):
                        self.samples.append((os.path.join(vid_dir, fname), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        if path.endswith(".npy"):
            tensor = np.load(path)
            return torch.from_numpy(tensor), label
        import cv2
        img = cv2.imread(path)
        if self.mode == "train":
            processed = preprocess_train(img)
        else:
            processed = preprocess_infer(img)
        return torch.from_numpy(processed), label


class VideoClipDataset(Dataset):
    """Loads frame sequences for temporal (CNN+LSTM) training."""

    def __init__(self, root: str, class_names: List[str],
                 seq_len: int = 16, stride: int = 4, mode: str = "train"):
        self.root = root
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.seq_len = seq_len
        self.stride = stride
        self.mode = mode
        self.clips: List[Tuple[str, int, int]] = []  # (video_dir, label, num_frames)
        self._scan(class_names)

    def _scan(self, class_names: List[str]):
        for cls_name in class_names:
            cls_dir = os.path.join(self.root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            label = self.class_to_idx[cls_name]
            for vid in sorted(os.listdir(cls_dir)):
                vid_dir = os.path.join(cls_dir, vid)
                if not os.path.isdir(vid_dir):
                    continue
                frames = sorted(f for f in os.listdir(vid_dir)
                                if f.endswith((".npy", ".jpg", ".png")))
                if len(frames) >= self.seq_len * self.stride:
                    self.clips.append((vid_dir, label, len(frames)))

    def __len__(self) -> int:
        return len(self.clips)

    def _load_frame(self, vid_dir: str, idx: int):
        frames = sorted(f for f in os.listdir(vid_dir)
                        if f.endswith((".npy", ".jpg", ".png")))
        idx = min(idx, len(frames) - 1)
        path = os.path.join(vid_dir, frames[idx])
        if path.endswith(".npy"):
            t = np.load(path)
            if t.ndim == 3 and t.shape[0] in (1, 3):
                return torch.from_numpy(t)
            return torch.from_numpy(np.transpose(t, (2, 0, 1)) / 255.0)
        import cv2
        img = cv2.imread(path)
        return torch.from_numpy(preprocess_infer(img))

    def __getitem__(self, idx: int):
        vid_dir, label, num_frames = self.clips[idx]
        max_start = max(0, num_frames - self.seq_len * self.stride)
        start = random.randint(0, max_start) if self.mode == "train" and max_start > 0 else 0
        indices = [min(start + i * self.stride, num_frames - 1) for i in range(self.seq_len)]
        clip = torch.stack([self._load_frame(vid_dir, i) for i in indices])  # (T, C, H, W)
        return clip, label


# ═══════════════════════════════════════════════════════════════════
# Metrics & Utilities
# ═══════════════════════════════════════════════════════════════════

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = 0.0; self.count = 0
    def update(self, val: float, n: int = 1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / max(self.count, 1)


class MetricsTracker:
    """Accumulates predictions for validation metrics."""

    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.preds, self.targets, self.probs = [], [], []

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        p = torch.softmax(outputs, dim=1).detach().cpu()
        self.preds.extend(p.argmax(dim=1).tolist())
        self.targets.extend(targets.cpu().tolist())
        self.probs.extend(p.numpy())

    def compute(self) -> Dict:
        from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                     roc_auc_score, confusion_matrix)
        y_true, y_pred = np.array(self.targets), np.array(self.preds)
        y_prob = np.array(self.probs)
        acc = accuracy_score(y_true, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        w_f1 = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]
        try:
            auc = roc_auc_score(np.eye(self.num_classes)[y_true], y_prob,
                                multi_class="ovr", average="macro")
        except ValueError:
            auc = 0.0
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        per_class = {self.class_names[i]: {"p": float(p[i]), "r": float(r[i]), "f1": float(f[i]),
                                            "support": int((y_true == i).sum())}
                      for i in range(self.num_classes)}
        return {"accuracy": float(acc), "macro_f1": float(f.mean()), "weighted_f1": float(w_f1),
                "auc": float(auc), "per_class": per_class, "confusion_matrix": cm}


class CheckpointManager:
    """Top-K checkpoint saving/loading."""

    def __init__(self, save_dir: str, top_k: int = 3):
        self.save_dir = save_dir
        self.top_k = top_k
        self.checkpoints: List[Tuple[float, str]] = []
        os.makedirs(save_dir, exist_ok=True)

    def save(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer,
             scheduler, metrics: Dict, is_best: bool = False):
        metric = metrics.get("macro_f1", 0.0)
        path = os.path.join(self.save_dir, f"epoch_{epoch:03d}_f1_{metric:.4f}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
        }, path)
        self.checkpoints.append((metric, path))
        if is_best:
            torch.save(model.state_dict(), os.path.join(self.save_dir, "best_model.pt"))
        self.checkpoints.sort(key=lambda x: x[0], reverse=True)
        while len(self.checkpoints) > self.top_k:
            _, old = self.checkpoints.pop()
            if os.path.exists(old) and "best_model" not in old:
                os.remove(old)

    def load_best(self) -> Optional[Dict]:
        p = os.path.join(self.save_dir, "best_model.pt")
        if os.path.exists(p):
            return torch.load(p, map_location="cpu", weights_only=False)
        return None


# ═══════════════════════════════════════════════════════════════════
# Loss & Scheduler
# ═══════════════════════════════════════════════════════════════════

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = pred.size(1)
        lp = torch.log_softmax(pred, dim=1)
        with torch.no_grad():
            st = torch.full_like(lp, self.smoothing / (n - 1))
            st.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return (-st * lp).sum(dim=1).mean()


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup: int, total: int, lr_min: float = 1e-6):
        self.warmup = warmup
        self.total = total
        self.lr_min = lr_min
        super().__init__(optimizer)
    def get_lr(self):
        e = self.last_epoch + 1
        if e <= self.warmup:
            return [b * e / self.warmup for b in self.base_lrs]
        p = min(1.0, (e - self.warmup) / max(1, self.total - self.warmup))
        return [self.lr_min + 0.5 * (b - self.lr_min) * (1.0 + math.cos(math.pi * p))
                for b in self.base_lrs]


# ═══════════════════════════════════════════════════════════════════
# Model factory
# ═══════════════════════════════════════════════════════════════════

def build_model(temporal: bool = False) -> nn.Module:
    if temporal:
        return TemporalModel(
            num_classes=CFG.num_classes, hidden_size=CFG.hidden_size,
            num_layers=CFG.num_lstm_layers, dropout=CFG.lstm_dropout,
            bidirectional=CFG.bidirectional, freeze_backbone=CFG.freeze_backbone,
        )
    return EfficientNetClassifier(
        num_classes=CFG.num_classes, dropout=CFG.dropout,
        freeze_backbone=CFG.freeze_backbone,
    )


# ═══════════════════════════════════════════════════════════════════
# DataLoaders
# ═══════════════════════════════════════════════════════════════════

def create_loaders(temporal: bool = False):
    ds = FaceFrameDataset(CFG.dataset_faces, CFG.classes, mode="train")
    if not temporal:
        indices = list(range(len(ds)))
        train_i, temp_i = train_test_split(indices, test_size=CFG.val_split + CFG.test_split,
                                           random_state=CFG.seed)
        val_i, test_i = train_test_split(temp_i,
            test_size=CFG.test_split / (CFG.val_split + CFG.test_split), random_state=CFG.seed)
        # Reuse already-scanned samples to avoid redundant filesystem traversal
        train_ds = FaceFrameDataset.__new__(FaceFrameDataset)
        train_ds.root = ds.root; train_ds.class_to_idx = ds.class_to_idx
        train_ds.mode = "train"; train_ds.samples = [ds.samples[i] for i in train_i]
        val_ds = FaceFrameDataset.__new__(FaceFrameDataset)
        val_ds.root = ds.root; val_ds.class_to_idx = ds.class_to_idx
        val_ds.mode = "val"; val_ds.samples = [ds.samples[i] for i in val_i]
        test_ds = FaceFrameDataset.__new__(FaceFrameDataset)
        test_ds.root = ds.root; test_ds.class_to_idx = ds.class_to_idx
        test_ds.mode = "val"; test_ds.samples = [ds.samples[i] for i in test_i]
    else:
        train_ds = VideoClipDataset(CFG.dataset_faces, CFG.classes, CFG.temporal_seq_len, mode="train")
        val_ds = VideoClipDataset(CFG.dataset_faces, CFG.classes, CFG.temporal_seq_len, mode="val")
        test_ds = val_ds  # Re-use val for test in temporal mode
    bs = CFG.temporal_batch_size if temporal else CFG.batch_size
    train_ld = DataLoader(train_ds, bs, shuffle=True, num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, bs, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)
    test_ld = DataLoader(test_ds, bs, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)
    return train_ld, val_ld, test_ld


# ═══════════════════════════════════════════════════════════════════
# Train / Validate Epochs
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, scaler, scheduler, epoch, writer, temporal, global_step=0):
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    pbar = tqdm(loader, desc=f"Train {epoch+1}", ncols=80, file=sys.stdout)
    for step, batch in enumerate(pbar):
        if temporal:
            x, y = batch; B, T, C, H, W = x.shape
            x = x.to(CFG.device, non_blocking=True)
        else:
            x, y = batch
            x = x.to(CFG.device, non_blocking=True)
        y = y.to(CFG.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=CFG.amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if CFG.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
        loss_m.update(loss.item(), x.size(0))
        acc_m.update(acc, x.size(0))
        pbar.set_postfix(loss=loss_m.avg, acc=acc_m.avg * 100)
        if step % 10 == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/acc", acc, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1
    if scheduler is not None:
        scheduler.step()
    return loss_m.avg, acc_m.avg, global_step


@torch.no_grad()
def val_epoch(model, loader, criterion, epoch, writer, tracker, temporal):
    model.eval()
    loss_m = AverageMeter()
    tracker.reset()
    pbar = tqdm(loader, desc=f"Val   {epoch+1}", ncols=80, file=sys.stdout)
    for batch in pbar:
        if temporal:
            x, y = batch; B, T, C, H, W = x.shape
            x = x.to(CFG.device, non_blocking=True)
        else:
            x, y = batch; x = x.to(CFG.device, non_blocking=True)
        y = y.to(CFG.device, non_blocking=True)
        with autocast(enabled=CFG.amp):
            logits = model(x)
            loss = criterion(logits, y)
        tracker.update(logits, y)
        loss_m.update(loss.item(), x.size(0))
        acc = (logits.argmax(dim=1) == y).float().mean().item()
        pbar.set_postfix(loss=loss_m.avg, acc=acc * 100)
    m = tracker.compute()
    writer.add_scalar("val/loss", loss_m.avg, epoch)
    writer.add_scalar("val/accuracy", m["accuracy"], epoch)
    writer.add_scalar("val/macro_f1", m["macro_f1"], epoch)
    writer.add_scalar("val/auc", m["auc"], epoch)
    return loss_m.avg, m


# ═══════════════════════════════════════════════════════════════════
# Entry: train()
# ═══════════════════════════════════════════════════════════════════

def train(temporal: bool = False, resume: Optional[str] = None):
    CFG.ensure_dirs()
    random.seed(CFG.seed); np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed); torch.cuda.manual_seed_all(CFG.seed)

    print("Loading data...")
    train_ld, val_ld, test_ld = create_loaders(temporal)
    M = len(train_ld.dataset); V = len(val_ld.dataset)
    print(f"  Train: {M}  Val: {V}  Classes: {CFG.classes}")

    model = build_model(temporal).to(CFG.device)
    print(f"  Model: {'Temporal' if temporal else 'EfficientNet-B4'}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = LabelSmoothingCE(CFG.label_smoothing) if CFG.label_smoothing > 0 else nn.CrossEntropyLoss()
    if CFG.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    elif CFG.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=CFG.learning_rate, momentum=CFG.momentum, weight_decay=CFG.weight_decay)

    scheduler = None
    if CFG.scheduler == "cosine":
        scheduler = CosineWarmupScheduler(optimizer, CFG.warmup_epochs, CFG.max_epochs, CFG.lr_min)
    elif CFG.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=CFG.lr_patience,
                                                          factor=CFG.lr_gamma, min_lr=CFG.lr_min)
    elif CFG.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, CFG.lr_step_size, CFG.lr_gamma)

    scaler = GradScaler(enabled=CFG.amp)

    run_dir = os.path.join(CFG.log_dir, f"{'temporal' if temporal else 'frame'}_{CFG.experiment_name}")
    writer = SummaryWriter(run_dir)
    ckpt_dir = os.path.join(CFG.checkpoint_dir, f"{'temporal' if temporal else 'frame'}_{CFG.experiment_name}")
    ckpt_mgr = CheckpointManager(ckpt_dir, CFG.save_top_k)

    start_epoch = 0
    if resume:
        print(f"Resuming: {resume}")
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") and scheduler:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1

    best_metric = -float("inf")
    stop_counter = 0
    tracker = MetricsTracker(CFG.num_classes, CFG.classes)

    print(f"\n{'='*50}")
    print(f"Training {'Temporal' if temporal else 'Frame-Level'} Model")
    print(f"Epochs: {CFG.max_epochs} | BS: {CFG.batch_size if not temporal else CFG.temporal_batch_size}")
    print(f"AMP: {CFG.amp} | LR: {CFG.learning_rate} | Scheduler: {CFG.scheduler}")
    print(f"{'='*50}\n")

    _step = 0  # TensorBoard global step counter
    for epoch in range(start_epoch, CFG.max_epochs):
        t0 = time.time()
        train_loss, train_acc, _step = train_epoch(model, train_ld, criterion, optimizer, scaler, scheduler, epoch, writer, temporal, global_step=_step)
        val_loss, val_m = val_epoch(model, val_ld, criterion, epoch, writer, tracker, temporal)
        dt = time.time() - t0

        print(f"Epoch {epoch+1:2d}/{CFG.max_epochs} | Train L:{train_loss:.4f} A:{train_acc*100:.2f}% | "
              f"Val L:{val_loss:.4f} A:{val_m['accuracy']*100:.2f}% F1:{val_m['macro_f1']:.4f} | "
              f"LR:{optimizer.param_groups[0]['lr']:.2e} | {dt:.0f}s")

        if CFG.scheduler == "plateau" and scheduler:
            scheduler.step(val_m["macro_f1"])

        is_best = val_m["macro_f1"] > best_metric
        if is_best:
            best_metric = val_m["macro_f1"]; stop_counter = 0
        else:
            stop_counter += 1
        ckpt_mgr.save(epoch, model, optimizer, scheduler, val_m, is_best)
        if stop_counter >= CFG.early_stop_patience:
            print(f"Early stop at epoch {epoch+1}"); break

    # ── Final test ──
    print(f"\n{'='*50}\nFinal Test Evaluation\n{'='*50}")
    best_state = ckpt_mgr.load_best()
    if best_state:
        model.load_state_dict(best_state)
        print("Loaded best model.")
    test_loss, test_m = val_epoch(model, test_ld, criterion, CFG.max_epochs, writer, tracker, temporal)
    print(f"Test Accuracy: {test_m['accuracy']:.4f}")
    print(f"Test Macro F1: {test_m['macro_f1']:.4f}")
    print(f"Test AUC:      {test_m['auc']:.4f}")
    for cls, cm in test_m["per_class"].items():
        print(f"  {cls:<12} P:{cm['p']:.4f} R:{cm['r']:.4f} F1:{cm['f1']:.4f}")
    writer.close()
    return test_m


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train deepfake detection model")
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()
    if args.epochs: CFG.max_epochs = args.epochs
    if args.batch_size: CFG.batch_size = args.batch_size
    if args.lr: CFG.learning_rate = args.lr
    train(temporal=args.temporal, resume=args.resume)
