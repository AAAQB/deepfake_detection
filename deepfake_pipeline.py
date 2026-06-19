"""
deepfake_pipeline.py — Unified Deepfake Detection Pipeline (Single File)
=========================================================================
Directory structure requirements:
    data_raw/
        image/
            real/       *.jpg / *.png / ...
            deepfake/
            filter/
        video/
            real/       *.mp4 / *.avi / *.mov
            deepfake/
            filter/

Preprocessing output:
    data_face/
        image/
            real/       {stem}/00000_0.npy ...
            deepfake/
            filter/
        video/
            real/       {stem}/00000_0.npy ...  (ordered by frame)
            deepfake/
            filter/

Training strategy:
    EfficientNet-B4  ← All image faces + 1 random frame per video
    LSTM (Temporal)  ← Sequential video frames sliced via rolling window

Final outputs:
    checkpoints/
        frame_*/best_model.pt      # EfficientNet weights
        temporal_*/best_model.pt   # EfficientNet+LSTM weights
        fused_model.pt             # Complete ensemble dictionary

Usage:
    # 1. Preprocessing only
    python deepfake_pipeline.py preprocess

    # 2. Train EfficientNet only
    python deepfake_pipeline.py train_frame

    # 3. Train LSTM only
    python deepfake_pipeline.py train_temporal

    # 4. End-to-end pipeline execution
    python deepfake_pipeline.py all
"""

# ======================================================================
# 0. Windows Multi-processing Safe Guard
# ======================================================================
import os as _os
_os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
_os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
_os.environ.setdefault("XNNPACK_FORCE_SINGLE_THREAD", "1")

# ======================================================================
# 1. Standard Imports
# ======================================================================
import os
import sys
import cv2
import math
import time
import random
import argparse
import multiprocessing
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False
    print("[WARN] timm not installed — EfficientNet will use a lightweight fallback.")

# ======================================================================
# 2. Global Configuration
# ======================================================================

@dataclass
class CFG:
    # ── Paths ─────────────────────────────────────────────────────────
    raw_root:        str = "data_raw"
    face_root:       str = "data_face"
    checkpoint_dir:  str = "checkpoints"
    log_dir:         str = "logs"
    experiment_name: str = "deepfake_v1"

    # ── Classes ───────────────────────────────────────────────────────
    classes:     List[str] = field(default_factory=lambda: ["real", "filter", "deepfake"])
    num_classes: int = 3

    # ── Preprocessing ─────────────────────────────────────────────────
    frame_interval: int  = 5      
    num_workers_pp: int  = 4      

    # ── Data Split ────────────────────────────────────────────────────
    train_split: float = 0.70
    val_split:   float = 0.15
    test_split:  float = 0.15
    seed:        int   = 42

    # ── EfficientNet Hyperparameters ──────────────────────────────────
    backbone:         str   = "efficientnet_b4"
    pretrained:       bool  = True
    dropout:          float = 0.4
    freeze_backbone:  bool  = False

    # ── LSTM Hyperparameters ──────────────────────────────────────────
    hidden_size:      int   = 512
    num_lstm_layers:  int   = 4
    lstm_dropout:     float = 0.4
    bidirectional:    bool  = True
    temporal_seq_len: int   = 16
    temporal_stride:  int   = 4

    # ── Training General ──────────────────────────────────────────────
    batch_size:          int   = 64
    temporal_batch_size: int   = 8
    num_workers_train:   int   = 2
    optimizer:           str   = "adamw"
    learning_rate:       float = 1e-4
    weight_decay:        float = 1e-4
    momentum:            float = 0.9
    scheduler:           str   = "cosine"
    lr_min:              float = 1e-6
    lr_patience:         int   = 5
    lr_step_size:        int   = 10
    lr_gamma:            float = 0.5
    warmup_epochs:       int   = 3
    label_smoothing:     float = 0.1
    gradient_clip_norm:  float = 1.0
    max_epochs:          int   = 40
    early_stop_patience: int   = 10
    save_top_k:          int   = 3
    amp:                 bool  = field(default_factory=lambda: torch.cuda.is_available())
    device:              str   = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def ensure_dirs(self):
        for d in [self.checkpoint_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)


C = CFG()


# ======================================================================
# 3. Preprocessing
# ======================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS    = {".mp4", ".avi", ".mov", ".mkv"}


def _normalize_face(face_bgr: np.ndarray) -> np.ndarray:
    """BGR face → CHW float32 (ImageNet normalized)."""
    face = cv2.resize(face_bgr, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    face = (face - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(face, (2, 0, 1)).astype(np.float32)


def _detect_and_save_faces(image_bgr: np.ndarray,
                            mp_face,
                            save_dir: str,
                            base_id: int) -> int:
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)
    count = 0
    if results.detections:
        for i, det in enumerate(results.detections):
            bb = det.location_data.relative_bounding_box
            y1 = max(0, int(bb.ymin * h))
            x1 = max(0, int(bb.xmin * w))
            y2 = min(h, int((bb.ymin + bb.height) * h))
            x2 = min(w, int((bb.xmin + bb.width) * w))
            face = image_bgr[y1:y2, x1:x2]
            if face.size == 0:
                continue
            npy = _normalize_face(face)
            np.save(os.path.join(save_dir, f"{base_id:05d}_{i}.npy"), npy)
            count += 1
    return count


# ── Image Preprocessing worker ────────────────────────────────────────

def _process_single_image(args: tuple):
    import mediapipe as mp
    img_path, save_dir = args
    os.makedirs(save_dir, exist_ok=True)

    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)

    img = cv2.imread(img_path)
    if img is None:
        print(f"[SKIP] Cannot read {img_path}", flush=True)
        mp_face.close()
        return

    _detect_and_save_faces(img, mp_face, save_dir, base_id=0)
    mp_face.close()
    print(f"[IMG Done] {img_path}", flush=True)


# ── Video Preprocessing worker ────────────────────────────────────────

def _process_single_video(args: tuple):
    import mediapipe as mp
    video_path, save_dir, frame_interval = args
    os.makedirs(save_dir, exist_ok=True)

    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved_id = 0
    print(f"[VID Processing] {video_path}", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            _detect_and_save_faces(frame, mp_face, save_dir, base_id=saved_id)
            saved_id += 1
        frame_id += 1

    cap.release()
    mp_face.close()
    print(f"[VID Done] {video_path}", flush=True)


# ── Main Preprocessing Manager ───────────────────────────────────────

def run_preprocessing():
    num_workers = C.num_workers_pp
    img_tasks   = []
    video_tasks = []

    for modality in ("image", "video"):
        for cls in C.classes:
            src_dir = os.path.join(C.raw_root, modality, cls)
            if not os.path.isdir(src_dir):
                print(f"[Skip] {src_dir} not found")
                continue
            for fname in os.listdir(src_dir):
                ext = os.path.splitext(fname)[1].lower()
                full_path = os.path.join(src_dir, fname)
                stem      = os.path.splitext(fname)[0]
                save_dir  = os.path.join(C.face_root, modality, cls, stem)

                if modality == "image" and ext in IMAGE_EXTS:
                    img_tasks.append((full_path, save_dir))
                elif modality == "video" and ext in VIDEO_EXTS:
                    video_tasks.append((full_path, save_dir, C.frame_interval))

    print(f"Image tasks: {len(img_tasks)}  Video tasks: {len(video_tasks)}  Workers: {num_workers}")

    if img_tasks:
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(_process_single_image, img_tasks)

    if video_tasks:
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(_process_single_video, video_tasks)

    print(f"\n[Preprocessing Finished] Output path: {C.face_root}")


# ======================================================================
# 4. Model Definitions
# ======================================================================

class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B4 feature backbone classification module.
    """
    def __init__(self, num_classes: int = 3, dropout: float = 0.4,
                 freeze_backbone: bool = False):
        super().__init__()
        if _TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                "efficientnet_b4", pretrained=C.pretrained, num_classes=0)
            feat_dim = self.backbone.num_features
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1))
            feat_dim = 32

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _TIMM_AVAILABLE:
            f = self.backbone(x)
        else:
            f = self.backbone(x).flatten(1)
        return self.head(f)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if _TIMM_AVAILABLE:
            return self.backbone(x)
        return self.backbone(x).flatten(1)


class TemporalModel(nn.Module):
    """
    EfficientNet-B4 Feature Extraction followed by Bidirectional LSTM.
    Input shape requirement: (B, T, C, H, W)
    """
    def __init__(self, num_classes: int = 3, hidden_size: int = 512,
                 num_layers: int = 4, dropout: float = 0.4,
                 bidirectional: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.cnn = EfficientNetClassifier(
            num_classes=num_classes, dropout=dropout,
            freeze_backbone=freeze_backbone)

        if _TIMM_AVAILABLE:
            feat_dim = self.cnn.backbone.num_features
        else:
            feat_dim = 32

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional)

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats   = self.cnn.extract_features(x_flat)          
        feats   = feats.view(B, T, -1)                        
        out, _  = self.lstm(feats)                            
        pooled  = out.mean(dim=1)                             
        return self.classifier(pooled)


class FusedModel(nn.Module):
    """
    Ensemble inference packaging wrapper for classification scoring.
    """
    def __init__(self, frame_model: EfficientNetClassifier,
                 temporal_model: TemporalModel,
                 frame_weight: float = 0.4):
        super().__init__()
        self.frame_model    = frame_model
        self.temporal_model = temporal_model
        self.frame_weight   = frame_weight

    @torch.no_grad()
    def forward_frame(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.frame_model(x), dim=1)

    @torch.no_grad()
    def forward_temporal(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.temporal_model(x), dim=1)

    @torch.no_grad()
    def forward(self, frame_x: torch.Tensor,
                temporal_x: torch.Tensor) -> torch.Tensor:
        p_frame    = self.forward_frame(frame_x)
        p_temporal = self.forward_temporal(temporal_x)
        return self.frame_weight * p_frame + (1 - self.frame_weight) * p_temporal


# ======================================================================
# 5. Datasets & DataLoaders
# ======================================================================

class FrameDataset(Dataset):
    def __init__(self,
                 image_items: List[Tuple[str, int]],
                 video_dirs:  List[Tuple[str, int]],
                 is_train: bool = False):
        self.image_items = image_items   
        self.video_dirs  = video_dirs    
        self.is_train    = is_train
        self.video_samples: List[Tuple[str, int]] = []
        self._sample_video_frames()

    def _sample_video_frames(self):
        samples = []
        for vid_dir, label in self.video_dirs:
            if not os.path.isdir(vid_dir):
                continue
            all_npy = sorted([f for f in os.listdir(vid_dir) if f.endswith(".npy")])
            if not all_npy:
                continue
            if self.is_train:
                chosen = random.choice(all_npy)
            else:
                chosen = all_npy[len(all_npy) // 2]
            samples.append((os.path.join(vid_dir, chosen), label))
        self.video_samples = samples

    def update_samples(self):
        if self.is_train:
            self._sample_video_frames()

    def __len__(self):
        return len(self.image_items) + len(self.video_samples)

    def __getitem__(self, idx: int):
        if idx < len(self.image_items):
            path, label = self.image_items[idx]
        else:
            path, label = self.video_samples[idx - len(self.image_items)]
        try:
            x = torch.from_numpy(np.load(path)).float()
        except Exception:
            x = torch.zeros((3, 224, 224), dtype=torch.float32)
        return x, torch.tensor(label, dtype=torch.long)


class TemporalDataset(Dataset):
    def __init__(self,
                 video_dirs: List[Tuple[str, int]],
                 seq_len: int = 16,
                 stride: int  = 4):
        self.clips: List[Tuple[List[str], int]] = []
        for vid_dir, label in video_dirs:
            if not os.path.isdir(vid_dir):
                continue
            frames = sorted([f for f in os.listdir(vid_dir) if f.endswith(".npy")])
            if len(frames) < seq_len:
                continue
            for start in range(0, len(frames) - seq_len + 1, stride):
                clip = [os.path.join(vid_dir, frames[start + i]) for i in range(seq_len)]
                self.clips.append((clip, label))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int):
        frame_paths, label = self.clips[idx]
        tensors = []
        for p in frame_paths:
            try:
                tensors.append(torch.from_numpy(np.load(p)).float())
            except Exception:
                tensors.append(torch.zeros((3, 224, 224), dtype=torch.float32))
        return torch.stack(tensors, dim=0), torch.tensor(label, dtype=torch.long)


def _scan_face_root() -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    class_to_idx = {c: i for i, c in enumerate(C.classes)}
    image_items: List[Tuple[str, int]] = []
    video_dirs:  List[Tuple[str, int]] = []

    for cls in C.classes:
        cls_dir = os.path.join(C.face_root, "image", cls)
        if not os.path.isdir(cls_dir):
            continue
        label = class_to_idx[cls]
        for stem in sorted(os.listdir(cls_dir)):
            stem_dir = os.path.join(cls_dir, stem)
            if not os.path.isdir(stem_dir):
                continue
            for fn in os.listdir(stem_dir):
                if fn.endswith(".npy"):
                    image_items.append((os.path.join(stem_dir, fn), label))

    for cls in C.classes:
        cls_dir = os.path.join(C.face_root, "video", cls)
        if not os.path.isdir(cls_dir):
            continue
        label = class_to_idx[cls]
        for stem in sorted(os.listdir(cls_dir)):
            stem_dir = os.path.join(cls_dir, stem)
            if os.path.isdir(stem_dir):
                video_dirs.append((stem_dir, label))

    return image_items, video_dirs


def create_frame_loaders():
    image_items, video_dirs = _scan_face_root()

    if not image_items and not video_dirs:
        raise RuntimeError("Processed face directory empty. Run preprocessing first.")

    vid_labels = [v[1] for v in video_dirs]
    if len(video_dirs) >= 3:
        train_vids, tmp_vids, _, tmp_labels = train_test_split(
            video_dirs, vid_labels,
            test_size=C.val_split + C.test_split,
            random_state=C.seed, stratify=vid_labels)
        val_vids, test_vids = train_test_split(
            tmp_vids,
            test_size=C.test_split / (C.val_split + C.test_split),
            random_state=C.seed + 1, stratify=tmp_labels)
    else:
        train_vids = video_dirs; val_vids = []; test_vids = []

    img_labels = [it[1] for it in image_items]
    if len(image_items) >= 3:
        train_imgs, tmp_imgs, _, tmp_img_labels = train_test_split(
            image_items, img_labels,
            test_size=C.val_split + C.test_split,
            random_state=C.seed, stratify=img_labels)
        val_imgs, test_imgs = train_test_split(
            tmp_imgs,
            test_size=C.test_split / (C.val_split + C.test_split),
            random_state=C.seed + 1, stratify=tmp_img_labels)
    else:
        train_imgs = image_items; val_imgs = []; test_imgs = []

    train_ds = FrameDataset(train_imgs, train_vids, is_train=True)
    val_ds   = FrameDataset(val_imgs,   val_vids,   is_train=False)
    test_ds  = FrameDataset(test_imgs,  test_vids,  is_train=False)

    pin  = C.device == "cuda"
    bs   = min(C.batch_size, max(1, len(train_ds)))
    drop = len(train_ds) > bs * 3

    train_ld = DataLoader(train_ds, bs, shuffle=True,  num_workers=C.num_workers_train,
                          pin_memory=pin, drop_last=drop)
    val_ld   = DataLoader(val_ds,  bs, shuffle=False, num_workers=C.num_workers_train,
                          pin_memory=pin)
    test_ld  = DataLoader(test_ds, bs, shuffle=False, num_workers=C.num_workers_train,
                          pin_memory=pin)
    return train_ld, val_ld, test_ld


def create_temporal_loaders():
    _, video_dirs = _scan_face_root()

    if not video_dirs:
        raise RuntimeError("Video features missing under data_face/video directory.")

    vid_labels = [v[1] for v in video_dirs]
    train_vids, tmp_vids, _, tmp_labels = train_test_split(
        video_dirs, vid_labels,
        test_size=C.val_split + C.test_split,
        random_state=C.seed, stratify=vid_labels)
    val_vids, test_vids = train_test_split(
        tmp_vids,
        test_size=C.test_split / (C.val_split + C.test_split),
        random_state=C.seed + 1, stratify=tmp_labels)

    train_ds = TemporalDataset(train_vids, C.temporal_seq_len, C.temporal_stride)
    val_ds   = TemporalDataset(val_vids,   C.temporal_seq_len, C.temporal_stride)
    test_ds  = TemporalDataset(test_vids,  C.temporal_seq_len, C.temporal_stride)

    pin = C.device == "cuda"
    bs  = C.temporal_batch_size

    train_ld = DataLoader(train_ds, bs, shuffle=True,  num_workers=C.num_workers_train,
                          pin_memory=pin, drop_last=True)
    val_ld   = DataLoader(val_ds,  bs, shuffle=False, num_workers=C.num_workers_train,
                          pin_memory=pin)
    test_ld  = DataLoader(test_ds, bs, shuffle=False, num_workers=C.num_workers_train,
                          pin_memory=pin)
    return train_ld, val_ld, test_ld


# ======================================================================
# 6. Training Utilities
# ======================================================================

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = 0.0; self.count = 0
    def update(self, val: float, n: int = 1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg  = self.sum / max(self.count, 1)


class MetricsTracker:
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes; self.class_names = class_names
        self.reset()

    def reset(self): self.preds, self.targets, self.probs = [], [], []

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        p = torch.softmax(outputs, dim=1).detach().cpu()
        self.preds.extend(p.argmax(dim=1).tolist())
        self.targets.extend(targets.cpu().tolist())
        self.probs.extend(p.numpy())

    def compute(self) -> Dict:
        from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                     roc_auc_score, confusion_matrix)
        y_true = np.array(self.targets); y_pred = np.array(self.preds)
        y_prob = np.array(self.probs)
        acc  = accuracy_score(y_true, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                      average=None, zero_division=0)
        w_f1 = precision_recall_fscore_support(y_true, y_pred,
                                               average="weighted", zero_division=0)[2]
        try:
            auc = roc_auc_score(np.eye(self.num_classes)[y_true], y_prob,
                                multi_class="ovr", average="macro")
        except ValueError:
            auc = 0.0
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        per_class = {
            self.class_names[i]: {"p": float(p[i]), "r": float(r[i]),
                                  "f1": float(f[i]),
                                  "support": int((y_true == i).sum())}
            for i in range(self.num_classes)}
        return {"accuracy": float(acc), "macro_f1": float(f.mean()),
                "weighted_f1": float(w_f1), "auc": float(auc),
                "per_class": per_class, "confusion_matrix": cm}


class CheckpointManager:
    def __init__(self, save_dir: str, top_k: int = 3):
        self.save_dir = save_dir; self.top_k = top_k
        self.checkpoints: List[Tuple[float, str]] = []
        os.makedirs(save_dir, exist_ok=True)

    def save(self, epoch: int, model: nn.Module, optimizer, scheduler,
             metrics: Dict, is_best: bool = False):
        metric = metrics.get("macro_f1", 0.0)
        ckpt = {"epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "metrics": metrics}
        path = os.path.join(self.save_dir, f"epoch_{epoch:03d}_f1_{metric:.4f}.pt")
        torch.save(ckpt, path)
        self.checkpoints.append((metric, path))
        if is_best:
            torch.save(ckpt, os.path.join(self.save_dir, "best_model.pt"))
        self.checkpoints.sort(key=lambda x: x[0], reverse=True)
        while len(self.checkpoints) > self.top_k:
            _, old = self.checkpoints.pop()
            if os.path.exists(old) and "best_model" not in old:
                os.remove(old)

    def load_best(self) -> Optional[Dict]:
        p = os.path.join(self.save_dir, "best_model.pt")
        if os.path.exists(p):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            return ck if "model_state_dict" in ck else {"model_state_dict": ck}
        return None


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__(); self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n  = pred.size(1)
        lp = torch.log_softmax(pred, dim=1)
        with torch.no_grad():
            st = torch.full_like(lp, self.smoothing / (n - 1))
            st.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return (-st * lp).sum(dim=1).mean()


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup: int, total: int, lr_min: float = 1e-6):
        self.warmup = warmup; self.total = total; self.lr_min = lr_min
        super().__init__(optimizer)

    def get_lr(self):
        e = self.last_epoch + 1
        if e <= self.warmup:
            return [b * e / self.warmup for b in self.base_lrs]
        prog = min(1.0, (e - self.warmup) / max(1, self.total - self.warmup))
        return [self.lr_min + 0.5 * (b - self.lr_min) * (1.0 + math.cos(math.pi * prog))
                for b in self.base_lrs]


def apply_gpu_augmentation(x: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated online augmentation tensor block pipeline operations."""
    B = x.shape[0]
    mask = torch.rand(B, device=x.device) > 0.5
    if mask.any():
        x = x.clone()
        x[mask] = torch.flip(x[mask], dims=[3])
    if random.random() > 0.7:
        x = x + torch.randn_like(x) * 0.03
    if random.random() > 0.5:
        angles = (torch.rand(B, device=x.device) * 12.0 - 6.0) * math.pi / 180.0
        scales  = torch.rand(B, device=x.device) * 0.08 + 0.96
        cos_a   = torch.cos(angles) * scales
        sin_a   = torch.sin(angles) * scales
        theta   = torch.zeros((B, 2, 3), device=x.device)
        theta[:, 0, 0] = cos_a;  theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a;  theta[:, 1, 1] = cos_a
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x    = F.grid_sample(x, grid, mode="bilinear",
                             padding_mode="reflection", align_corners=False)
    return x


# ======================================================================
# 7. Train / Validate Epoch
# ======================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, scheduler,
                epoch: int, writer: SummaryWriter, temporal: bool,
                global_step: int = 0) -> Tuple[float, float, int]:
    model.train()
    loss_m = AverageMeter(); acc_m = AverageMeter()
    pbar   = tqdm(loader, desc=f"Train {epoch + 1}", ncols=90, file=sys.stdout)

    if not temporal and hasattr(loader.dataset, "update_samples"):
        loader.dataset.update_samples()

    for step, (x, y) in enumerate(pbar):
        x = x.to(C.device, non_blocking=True)
        y = y.to(C.device, non_blocking=True)

        if not temporal:
            x = apply_gpu_augmentation(x)
        else:
            B, T, Cc, H, W = x.shape
            x_flat = x.view(B * T, Cc, H, W)
            x_aug  = apply_gpu_augmentation(x_flat)
            x      = x_aug.view(B, T, Cc, H, W)

        optimizer.zero_grad(set_to_none=True)

        if C.amp:
            with torch.amp.autocast("cuda"):
                logits = model(x); loss = criterion(logits, y)
        else:
            logits = model(x); loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if C.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), C.gradient_clip_norm)
        scaler.step(optimizer); scaler.update()

        acc = (logits.argmax(dim=1) == y).float().mean().item()
        loss_m.update(loss.item(), x.size(0)); acc_m.update(acc, x.size(0))
        pbar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg * 100:.2f}%")

        if step % 10 == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/acc", acc, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

    return loss_m.avg, acc_m.avg, global_step


@torch.no_grad()
def val_epoch(model, loader, criterion, epoch: int, writer: SummaryWriter,
              tracker: MetricsTracker, split: str = "val") -> Tuple[float, Dict]:
    model.eval()
    loss_m = AverageMeter(); tracker.reset()
    pbar = tqdm(loader, desc=f"{split:<5} {epoch + 1}", ncols=90, file=sys.stdout)

    for x, y in pbar:
        x = x.to(C.device, non_blocking=True)
        y = y.to(C.device, non_blocking=True)
        if C.amp:
            with torch.amp.autocast("cuda"):
                logits = model(x); loss = criterion(logits, y)
        else:
            logits = model(x); loss = criterion(logits, y)
        tracker.update(logits, y)
        loss_m.update(loss.item(), x.size(0))
        pbar.set_postfix(loss=f"{loss_m.avg:.4f}")

    m = tracker.compute()
    prefix = "val" if split == "val" else "test"
    writer.add_scalar(f"{prefix}/loss",      loss_m.avg,    epoch)
    writer.add_scalar(f"{prefix}/accuracy",  m["accuracy"], epoch)
    writer.add_scalar(f"{prefix}/macro_f1",  m["macro_f1"], epoch)
    writer.add_scalar(f"{prefix}/auc",       m["auc"],      epoch)
    return loss_m.avg, m


# ======================================================================
# 8. Main Training Loop & History Export
# ======================================================================

def _build_optimizer(model: nn.Module):
    if C.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=C.learning_rate,
                           weight_decay=C.weight_decay)
    if C.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=C.learning_rate,
                          weight_decay=C.weight_decay)
    return optim.SGD(model.parameters(), lr=C.learning_rate,
                     momentum=C.momentum, weight_decay=C.weight_decay)


def _build_scheduler(optimizer):
    if C.scheduler == "cosine":
        return CosineWarmupScheduler(optimizer, C.warmup_epochs, C.max_epochs, C.lr_min)
    if C.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=C.lr_patience,
            factor=C.lr_gamma, min_lr=C.lr_min)
    if C.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, C.lr_step_size, C.lr_gamma)
    return None


import matplotlib.pyplot as plt

def plot_training_history(history: Dict[str, List[float]], save_path: str, title: str):
    """Generates standard training curve performance trends mapping plots."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2, linestyle="--")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.set_title(f"{title} - Loss Target", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.plot([], [], ' ', label=f"Best Val F1: {max(history['val_f1']):.4f}")
    ax1.legend(frameon=True)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="#2ca02c", linewidth=2, linestyle="--")
    ax2.plot(epochs, history["val_acc"], label="Val Acc", color="#d62728", linewidth=2)
    ax2.set_title(f"{title} - Accuracy Target", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epochs", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[📊 Plot Exported] Performance curves saved to: {save_path}")


def train_model(temporal: bool = False,
                resume: Optional[str] = None) -> Dict:
    C.ensure_dirs()
    random.seed(C.seed); np.random.seed(C.seed)
    torch.manual_seed(C.seed); torch.cuda.manual_seed_all(C.seed)

    tag = "temporal" if temporal else "frame"
    print(f"\n{'=' * 60}")
    print(f"  Execution Mode: {'Temporal (EfficientNet+LSTM)' if temporal else 'Frame (EfficientNet-B4)'}")
    print(f"{'=' * 60}")

    print("Loading datasets...")
    if temporal:
        train_ld, val_ld, test_ld = create_temporal_loaders()
    else:
        train_ld, val_ld, test_ld = create_frame_loaders()
    print(f"  Train={len(train_ld.dataset)}  Val={len(val_ld.dataset)}  Test={len(test_ld.dataset)}")

    if temporal:
        model = TemporalModel(num_classes=C.num_classes,
                              hidden_size=C.hidden_size,
                              num_layers=C.num_lstm_layers,
                              dropout=C.lstm_dropout,
                              bidirectional=C.bidirectional,
                              freeze_backbone=C.freeze_backbone)
    else:
        model = EfficientNetClassifier(num_classes=C.num_classes,
                                       dropout=C.dropout,
                                       freeze_backbone=C.freeze_backbone)
    model = model.to(C.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model Parameters Count: {total_params:,}")

    criterion = (LabelSmoothingCE(C.label_smoothing) if C.label_smoothing > 0 else nn.CrossEntropyLoss())
    optimizer = _build_optimizer(model)
    scheduler = _build_scheduler(optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=C.amp) if C.device == "cuda" else torch.amp.GradScaler("cpu", enabled=False)

    run_dir = os.path.join(C.log_dir, f"{tag}_{C.experiment_name}")
    ckpt_dir = os.path.join(C.checkpoint_dir, f"{tag}_{C.experiment_name}")
    writer = SummaryWriter(run_dir)
    ckpt_mgr = CheckpointManager(ckpt_dir, C.save_top_k)
    tracker = MetricsTracker(C.num_classes, C.classes)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": []
    }

    start_epoch = 0
    if resume and os.path.exists(resume):
        print(f"Resuming model parameters checkpoint execution: {resume}")
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if scheduler and ck.get("scheduler_state_dict"):
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1

    best_metric = -float("inf")
    stop_counter = 0
    g_step = 0

    print(f"  Epochs={C.max_epochs} | BS={'temporal: ' + str(C.temporal_batch_size) if temporal else C.batch_size}")
    print(f"  AMP={C.amp} | LR={C.learning_rate} | Scheduler={C.scheduler}\n")

    for epoch in range(start_epoch, C.max_epochs):
        t0 = time.time()
        tr_loss, tr_acc, g_step = train_epoch(
            model, train_ld, criterion, optimizer, scaler,
            scheduler, epoch, writer, temporal, g_step)
        val_loss, val_m = val_epoch(
            model, val_ld, criterion, epoch, writer, tracker, split="val")
        dt = time.time() - t0

        print(f"Epoch {epoch + 1:2d}/{C.max_epochs} | "
              f"Train L:{tr_loss:.4f} A:{tr_acc * 100:.2f}% | "
              f"Val L:{val_loss:.4f} A:{val_m['accuracy'] * 100:.2f}% "
              f"F1:{val_m['macro_f1']:.4f} | "
              f"LR:{optimizer.param_groups[0]['lr']:.2e} | {dt:.0f}s")

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_m["accuracy"])
        history["val_f1"].append(val_m["macro_f1"])

        if C.scheduler == "plateau" and scheduler:
            scheduler.step(val_m["macro_f1"])
        elif scheduler:
            scheduler.step()

        is_best = val_m["macro_f1"] > best_metric
        if is_best:
            best_metric = val_m["macro_f1"]
            stop_counter = 0
        else:
            stop_counter += 1
        ckpt_mgr.save(epoch, model, optimizer, scheduler, val_m, is_best)

        if stop_counter >= C.early_stop_patience:
            print(f"  Early stopping triggered at epoch {epoch + 1}")
            break

    if len(history["train_loss"]) > 0:
        plot_save_path = os.path.join(ckpt_dir, f"{tag}_training_metrics.png")
        plot_title = "Temporal Model (LSTM)" if temporal else "Frame Model (EfficientNet)"
        plot_training_history(history, plot_save_path, plot_title)

    print(f"\n{'=' * 60}\nFinal Test Evaluation\n{'=' * 60}")
    best_state = ckpt_mgr.load_best()
    if best_state:
        model.load_state_dict(best_state["model_state_dict"])
        print("Loaded optimal parameters weights state.")
    test_loss, test_m = val_epoch(
        model, test_ld, criterion, C.max_epochs, writer, tracker, split="test")
    print(f"Test Accuracy: {test_m['accuracy']:.4f}")
    print(f"Test Macro F1: {test_m['macro_f1']:.4f}")
    print(f"Test AUC:      {test_m['auc']:.4f}")
    for cls, cm in test_m["per_class"].items():
        print(f"  {cls:<12} P:{cm['p']:.4f} R:{cm['r']:.4f} F1:{cm['f1']:.4f}")
    writer.close()
    return test_m

# ======================================================================
# 9. Fuse & Export
# ======================================================================

def fuse_and_export():
    frame_ckpt_dir    = os.path.join(C.checkpoint_dir, f"frame_{C.experiment_name}")
    temporal_ckpt_dir = os.path.join(C.checkpoint_dir, f"temporal_{C.experiment_name}")

    frame_best    = os.path.join(frame_ckpt_dir,    "best_model.pt")
    temporal_best = os.path.join(temporal_ckpt_dir, "best_model.pt")

    missing = [p for p in (frame_best, temporal_best) if not os.path.exists(p)]
    if missing:
        print("[WARN] Missing target weights state checkpoint parameters:")
        for m in missing:
            print(f"  {m}")
        return

    frame_state    = torch.load(frame_best,    map_location="cpu", weights_only=False)
    temporal_state = torch.load(temporal_best, map_location="cpu", weights_only=False)

    fused_path = os.path.join(C.checkpoint_dir, "fused_model.pt")
    torch.save({
        "frame_model_state_dict":    frame_state["model_state_dict"],
        "temporal_model_state_dict": temporal_state["model_state_dict"],
        "frame_metrics":    frame_state.get("metrics", {}),
        "temporal_metrics": temporal_state.get("metrics", {}),
        "config": {
            "num_classes":    C.num_classes,
            "classes":        C.classes,
            "hidden_size":    C.hidden_size,
            "num_lstm_layers":C.num_lstm_layers,
            "lstm_dropout":   C.lstm_dropout,
            "bidirectional":  C.bidirectional,
            "dropout":        C.dropout,
        }
    }, fused_path)
    print(f"\n[Ensemble Export Finished] Model saved to: {fused_path}")
    print(f"  Frame    Best F1: {frame_state.get('metrics', {}).get('macro_f1', 'N/A')}")
    print(f"  Temporal Best F1: {temporal_state.get('metrics', {}).get('macro_f1', 'N/A')}")


def load_fused_model(fused_path: str = None, device: str = "cpu"):
    if fused_path is None:
        fused_path = os.path.join(C.checkpoint_dir, "fused_model.pt")

    ckpt = torch.load(fused_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})

    frame_model = EfficientNetClassifier(
        num_classes=cfg.get("num_classes", C.num_classes),
        dropout=cfg.get("dropout", C.dropout))
    frame_model.load_state_dict(ckpt["frame_model_state_dict"])

    temporal_model = TemporalModel(
        num_classes=cfg.get("num_classes", C.num_classes),
        hidden_size=cfg.get("hidden_size", C.hidden_size),
        num_layers=cfg.get("num_lstm_layers", C.num_lstm_layers),
        dropout=cfg.get("lstm_dropout", C.lstm_dropout),
        bidirectional=cfg.get("bidirectional", C.bidirectional))
    temporal_model.load_state_dict(ckpt["temporal_model_state_dict"])

    fused = FusedModel(frame_model, temporal_model)
    fused.eval().to(device)
    print(f"[FusedModel] Initialized state loaded from {fused_path}")
    return fused


# ======================================================================
# 10. CLI Entry Point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Detection — Unified Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("command", choices=["preprocess", "train_frame",
                                            "train_temporal", "fuse", "all"],
                        help="Command execution phase assignment")
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch-size",  type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--resume",      type=str,   default=None)
    parser.add_argument("--raw-root",    type=str,   default=None)
    parser.add_argument("--face-root",   type=str,   default=None)
    parser.add_argument("--workers-pp",  type=int,   default=None)
    args = parser.parse_args()

    if args.epochs:     C.max_epochs    = args.epochs
    if args.batch_size: C.batch_size    = args.batch_size
    if args.lr:         C.learning_rate = args.lr
    if args.raw_root:   C.raw_root      = args.raw_root
    if args.face_root:  C.face_root     = args.face_root
    if args.workers_pp: C.num_workers_pp = args.workers_pp

    cmd = args.command

    if cmd == "preprocess":
        run_preprocessing()

    elif cmd == "train_frame":
        train_model(temporal=False, resume=args.resume)

    elif cmd == "train_temporal":
        train_model(temporal=True, resume=args.resume)

    elif cmd == "fuse":
        fuse_and_export()

    elif cmd == "all":
        print("Step 1: Preprocessing Executing")
        run_preprocessing()
        print("\nStep 2: Training EfficientNet (Frame Classifier)")
        train_model(temporal=False)
        print("\nStep 3: Training LSTM (Temporal Sequence Classifier)")
        train_model(temporal=True)
        print("\nStep 4: Model Fusion Compilation Ensemble")
        fuse_and_export()
        print("\nUnified Execution Pipeline Completed")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()