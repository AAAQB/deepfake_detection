"""
train.py — Optimized and Secured Training pipeline for deepfake detection models.
Fixes hidden validation leakage, data normalization, and single-frame redundancy.
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from EfficientNet import EfficientNetClassifier
from LSTM import TemporalModel


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class TrainingConfig:
    """Central training hyperparameters."""
    dataset_faces: str = "dataset_image_face"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    experiment_name: str = "deepfake_detection_v2"

    classes: List[str] = field(default_factory=lambda: ["real", "filter", "deepfake"])
    num_classes: int = 3

    batch_size: int = 64
    temporal_batch_size: int = 8
    num_workers: int = 2
    temporal_seq_len: int = 16
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42

    backbone: str = "efficientnet_b4"
    pretrained: bool = True
    dropout: float = 0.4
    freeze_backbone: bool = False
    hidden_size: int = 512
    num_lstm_layers: int = 4
    lstm_dropout: float = 0.4
    bidirectional: bool = True

    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9

    scheduler: str = "cosine"
    lr_min: float = 1e-6
    lr_patience: int = 5
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    warmup_epochs: int = 3

    label_smoothing: float = 0.1
    gradient_clip_norm: float = 1.0
    max_epochs: int = 40
    early_stop_patience: int = 10
    amp: bool = torch.cuda.is_available()
    save_top_k: int = 3

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_dirs(self):
        for d in [self.checkpoint_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)


CFG = TrainingConfig()


# =====================================================================
# Metrics & Utilities
# =====================================================================

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class MetricsTracker:
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
            auc = roc_auc_score(np.eye(self.num_classes)[y_true], y_prob, multi_class="ovr", average="macro")
        except ValueError:
            auc = 0.0
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        per_class = {self.class_names[i]: {"p": float(p[i]), "r": float(r[i]), "f1": float(f[i]),
                                           "support": int((y_true == i).sum())}
                     for i in range(self.num_classes)}
        return {"accuracy": float(acc), "macro_f1": float(f.mean()), "weighted_f1": float(w_f1),
                "auc": float(auc), "per_class": per_class, "confusion_matrix": cm}


class CheckpointManager:
    def __init__(self, save_dir: str, top_k: int = 3):
        self.save_dir = save_dir
        self.top_k = top_k
        self.checkpoints: List[Tuple[float, str]] = []
        os.makedirs(save_dir, exist_ok=True)

    def save(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer,
             scheduler, metrics: Dict, is_best: bool = False):
        metric = metrics.get("macro_f1", 0.0)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
        }
        path = os.path.join(self.save_dir, f"epoch_{epoch:03d}_f1_{metric:.4f}.pt")
        torch.save(checkpoint, path)
        self.checkpoints.append((metric, path))
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, "best_model.pt"))
        self.checkpoints.sort(key=lambda x: x[0], reverse=True)
        while len(self.checkpoints) > self.top_k:
            _, old = self.checkpoints.pop()
            if os.path.exists(old) and "best_model" not in old:
                os.remove(old)

    def load_best(self) -> Optional[Dict]:
        p = os.path.join(self.save_dir, "best_model.pt")
        if os.path.exists(p):
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            if "model_state_dict" not in ckpt:
                return {"model_state_dict": ckpt}
            return ckpt
        return None


# =====================================================================
# Loss & Scheduler
# =====================================================================

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


# =====================================================================
# Model factory
# =====================================================================

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


# =====================================================================
# Local Dataset & Loaders
# =====================================================================

class LocalDynamicFrameDataset(Dataset):
    """
    Dynamic single-frame dataset. Extracts one frame per video directory 
    to mitigate redundancy and prevent single-frame overfitting.
    """

    def __init__(self, video_dirs: List[Tuple[str, int]], is_train: bool = False):
        self.video_dirs = video_dirs
        self.is_train = is_train
        self.samples = []
        self.update_samples()

    def update_samples(self):
        """
        Dynamically samples frames at the beginning of each training epoch.
        """
        samples = []
        for vid_dir, label in self.video_dirs:
            if not os.path.isdir(vid_dir):
                continue
            all_npy = sorted([f for f in os.listdir(vid_dir) if f.endswith(".npy")])
            if len(all_npy) == 0:
                continue

            if self.is_train:
                # Randomly samples a different frame each epoch for data diversity
                chosen_fn = random.choice(all_npy)
            else:
                # Anchors to the middle frame for evaluation consistency
                chosen_fn = all_npy[len(all_npy) // 2]

            samples.append((os.path.join(vid_dir, chosen_fn), label))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            x = np.load(path)
            x = torch.from_numpy(x).float()
        except Exception:
            x = torch.zeros((3, 224, 224), dtype=torch.float32)
        return x, torch.tensor(label, dtype=torch.long)


class LocalTemporalDataset(Dataset):
    """
    Sequence dataset using video-level strict isolation.
    """

    def __init__(self, video_dirs: List[Tuple[str, int]], seq_len: int = 16, stride: int = 4, is_train: bool = False):
        self.clips = []
        self.is_train = is_train
        self.seq_len = seq_len

        for vid_dir, label in video_dirs:
            if not os.path.isdir(vid_dir):
                continue
            frames = sorted([f for f in os.listdir(vid_dir) if f.endswith(".npy")])
            if len(frames) < seq_len:
                continue

            # Sliding window segmentation
            for start_idx in range(0, len(frames) - seq_len + 1, stride):
                clip_frames = [os.path.join(vid_dir, frames[start_idx + i]) for i in range(seq_len)]
                self.clips.append((clip_frames, label))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        frame_paths, label = self.clips[idx]
        tensor_list = []
        for path in frame_paths:
            try:
                x = np.load(path)
                tensor_list.append(torch.from_numpy(x).float())
            except Exception:
                tensor_list.append(torch.zeros((3, 224, 224), dtype=torch.float32))
        return torch.stack(tensor_list, dim=0), torch.tensor(label, dtype=torch.long)


def create_loaders(temporal: bool = False):
    """
    Handles video-level dataset splitting and returns dataloaders.
    """
    class_to_idx = {c: i for i, c in enumerate(CFG.classes)}
    video_list = []

    # 1. Scan directories
    for cls_name in CFG.classes:
        cls_dir = os.path.join(CFG.dataset_faces, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        label = class_to_idx[cls_name]
        for vid in sorted(os.listdir(cls_dir)):
            vid_dir = os.path.join(cls_dir, vid)
            if os.path.isdir(vid_dir):
                video_list.append((vid_dir, label))

    if len(video_list) == 0:
        raise RuntimeError(f"No video folders found under {CFG.dataset_faces}")

    # 2. Split with strict video-level isolation
    video_labels = [v[1] for v in video_list]
    train_vids, temp_vids, _, temp_labels = train_test_split(
        video_list, video_labels,
        test_size=CFG.val_split + CFG.test_split,
        random_state=CFG.seed, stratify=video_labels
    )
    val_vids, test_vids = train_test_split(
        temp_vids,
        test_size=CFG.test_split / (CFG.val_split + CFG.test_split),
        random_state=CFG.seed + 1, stratify=temp_labels
    )

    if not temporal:
        train_ds = LocalDynamicFrameDataset(train_vids, is_train=True)
        val_ds = LocalDynamicFrameDataset(val_vids, is_train=False)
        test_ds = LocalDynamicFrameDataset(test_vids, is_train=False)
        bs = min(CFG.batch_size, len(train_ds))
    else:
        train_ds = LocalTemporalDataset(train_vids, seq_len=CFG.temporal_seq_len, is_train=True)
        val_ds = LocalTemporalDataset(val_vids, seq_len=CFG.temporal_seq_len, is_train=False)
        test_ds = LocalTemporalDataset(test_vids, seq_len=CFG.temporal_seq_len, is_train=False)
        bs = CFG.temporal_batch_size

    pin = CFG.device == "cuda"
    drop_last = len(train_ds) > bs * 3
    train_ld = DataLoader(train_ds, bs, shuffle=True, num_workers=CFG.num_workers, pin_memory=pin, drop_last=drop_last)
    val_ld = DataLoader(val_ds, bs, shuffle=False, num_workers=CFG.num_workers, pin_memory=pin)
    test_ld = DataLoader(test_ds, bs, shuffle=False, num_workers=CFG.num_workers, pin_memory=pin)
    return train_ld, val_ld, test_ld


# =====================================================================
# GPU Online Augmentation Pipeline
# =====================================================================

def apply_gpu_augmentation(x: torch.Tensor) -> torch.Tensor:
    """
    Applies real-time geometric and noise augmentations 
    directly onto the normalized GPU tensor space.
    """
    B, C, H, W = x.shape

    # 1. Random horizontal flip (50% probability)
    flip_mask = torch.rand(B, device=x.device) > 0.5
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])

    # 2. Inject slight Gaussian noise to counteract codec artifacts overfitting
    if random.random() > 0.7:
        noise = torch.randn_like(x) * 0.03
        x = x + noise

    # 3. Affine transformations: slight random rotation and scaling
    if random.random() > 0.5:
        angles = (torch.rand(B, device=x.device) * 12.0 - 6.0) * math.pi / 180.0
        scales = torch.rand(B, device=x.device) * 0.08 + 0.96

        cos_a = torch.cos(angles) * scales
        sin_a = torch.sin(angles) * scales

        theta = torch.zeros((B, 2, 3), device=x.device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)

    return x


# =====================================================================
# Train / Validate Epochs
# =====================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, scheduler, epoch, writer, temporal, global_step=0):
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    pbar = tqdm(loader, desc=f"Train {epoch + 1}", ncols=80, file=sys.stdout)

    # Reshuffle/sample frame mappings each epoch for single-frame dataset configurations
    if not temporal and hasattr(loader.dataset, 'update_samples'):
        loader.dataset.update_samples()

    for step, batch in enumerate(pbar):
        x, y = batch
        x = x.to(CFG.device, non_blocking=True)
        y = y.to(CFG.device, non_blocking=True)

        # Apply real-time augmentations depending on input dimensionality
        if not temporal:
            x = apply_gpu_augmentation(x)
        else:
            B, S, C, H, W = x.shape
            x_reshaped = x.view(B * S, C, H, W)
            x_aug = apply_gpu_augmentation(x_reshaped)
            x = x_aug.view(B, S, C, H, W)

        optimizer.zero_grad(set_to_none=True)

        if CFG.amp:
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(logits, y)
        else:
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

    return loss_m.avg, acc_m.avg, global_step


@torch.no_grad()
def val_epoch(model, loader, criterion, epoch, writer, tracker, temporal):
    model.eval()
    loss_m = AverageMeter()
    tracker.reset()
    pbar = tqdm(loader, desc=f"Val   {epoch + 1}", ncols=80, file=sys.stdout)
    for batch in pbar:
        x, y = batch
        x = x.to(CFG.device, non_blocking=True)
        y = y.to(CFG.device, non_blocking=True)

        if CFG.amp:
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(logits, y)
        else:
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


# =====================================================================
# Entry: train()
# =====================================================================

def train(temporal: bool = False, resume: Optional[str] = None):
    CFG.ensure_dirs()
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    torch.cuda.manual_seed_all(CFG.seed)

    print("Loading data...")
    train_ld, val_ld, test_ld = create_loaders(temporal)
    print(f"  Train size (Units): {len(train_ld.dataset)}  Val: {len(val_ld.dataset)}  Classes: {CFG.classes}")

    model = build_model(temporal).to(CFG.device)
    print(f"  Model Architecture: {'Temporal (EfficientNet+LSTM)' if temporal else 'EfficientNet-B4'}")
    print(f"  Params Count: {sum(p.numel() for p in model.parameters()):,}")

    criterion = LabelSmoothingCE(CFG.label_smoothing) if CFG.label_smoothing > 0 else nn.CrossEntropyLoss()
    if CFG.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    elif CFG.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=CFG.learning_rate, momentum=CFG.momentum,
                              weight_decay=CFG.weight_decay)

    scheduler = None
    if CFG.scheduler == "cosine":
        scheduler = CosineWarmupScheduler(optimizer, CFG.warmup_epochs, CFG.max_epochs, CFG.lr_min)
    elif CFG.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=CFG.lr_patience,
                                                         factor=CFG.lr_gamma, min_lr=CFG.lr_min)
    elif CFG.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, CFG.lr_step_size, CFG.lr_gamma)

    scaler = torch.amp.GradScaler('cuda', enabled=CFG.amp) if CFG.device == "cuda" else torch.amp.GradScaler('cpu',
                                                                                                             enabled=False)

    run_dir = os.path.join(CFG.log_dir, f"{'temporal' if temporal else 'frame'}_{CFG.experiment_name}")
    writer = SummaryWriter(run_dir)
    ckpt_dir = os.path.join(CFG.checkpoint_dir, f"{'temporal' if temporal else 'frame'}_{CFG.experiment_name}")
    ckpt_mgr = CheckpointManager(ckpt_dir, CFG.save_top_k)

    start_epoch = 0
    if resume:
        print(f"Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") and scheduler:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1

    best_metric = -float("inf")
    stop_counter = 0
    tracker = MetricsTracker(CFG.num_classes, CFG.classes)

    print(f"\n{'=' * 50}")
    print(f"Standardized Deepfake Pipeline Active")
    print(f"Epochs: {CFG.max_epochs} | BS: {CFG.batch_size if not temporal else CFG.temporal_batch_size}")
    print(f"AMP: {CFG.amp} | LR: {CFG.learning_rate} | Scheduler: {CFG.scheduler}")
    print(f"{'=' * 50}\n")

    _step = 0
    for epoch in range(start_epoch, CFG.max_epochs):
        t0 = time.time()
        train_loss, train_acc, _step = train_epoch(model, train_ld, criterion, optimizer, scaler, scheduler, epoch,
                                                   writer, temporal, global_step=_step)
        val_loss, val_m = val_epoch(model, val_ld, criterion, epoch, writer, tracker, temporal)
        dt = time.time() - t0

        print(f"Epoch {epoch + 1:2d}/{CFG.max_epochs} | Train L:{train_loss:.4f} A:{train_acc * 100:.2f}% | "
              f"Val L:{val_loss:.4f} A:{val_m['accuracy'] * 100:.2f}% F1:{val_m['macro_f1']:.4f} | "
              f"LR:{optimizer.param_groups[0]['lr']:.2e} | {dt:.0f}s")

        if CFG.scheduler == "plateau" and scheduler:
            scheduler.step(val_m["macro_f1"])
        elif scheduler is not None:
            scheduler.step()

        is_best = val_m["macro_f1"] > best_metric
        if is_best:
            best_metric = val_m["macro_f1"]
            stop_counter = 0
        else:
            stop_counter += 1
        ckpt_mgr.save(epoch, model, optimizer, scheduler, val_m, is_best)
        if stop_counter >= CFG.early_stop_patience:
            print(f"Early stop triggered at epoch {epoch + 1}")
            break

    print(f"\n{'=' * 50}\nFinal Generalization Test Evaluation\n{'=' * 50}")
    best_state = ckpt_mgr.load_best()
    if best_state:
        model.load_state_dict(best_state["model_state_dict"])
        print("Loaded best model state across all epochs.")
    test_loss, test_m = val_epoch(model, test_ld, criterion, CFG.max_epochs, writer, tracker, temporal)
    print(f"Test Accuracy: {test_m['accuracy']:.4f}")
    print(f"Test Macro F1: {test_m['macro_f1']:.4f}")
    print(f"Test AUC:      {test_m['auc']:.4f}")
    for cls, cm in test_m["per_class"].items():
        print(f"  {cls:<12} P:{cm['p']:.4f} R:{cm['r']:.4f} F1:{cm['f1']:.4f}")
    writer.close()
    return test_m


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
