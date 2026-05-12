"""
interface.py — Data processing, inference pipeline, model export, and ensemble.

Covers:
  - Data augmentation (colour, geometric, noise, compression, cutout)
  - PyTorch Dataset loading (frame-level + temporal)
  - Face detection & extraction via MediaPipe
  - Real-time inference (webcam, video file, single image)
  - Visualisation (bounding boxes, confidence bars, per-class panel)
  - Model export (ONNX, TorchScript)
  - Weighted ensemble inference
"""

import os
import sys
import time
import argparse
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from preprocessing import preprocess_train, preprocess_infer
from EfficientNet import EfficientNetClassifier
from LSTM import TemporalModel


# ═══════════════════════════════════════════════════════════════════
# I. DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════

def random_hflip(img: np.ndarray, p: float = 0.5) -> np.ndarray:
    return cv2.flip(img, 1) if np.random.rand() < p else img


def random_brightness(img: np.ndarray, factor: float = 0.2) -> np.ndarray:
    a = 1.0 + np.random.uniform(-factor, factor)
    return np.clip(img.astype(np.float32) * a, 0, 255).astype(np.uint8)


def random_contrast(img: np.ndarray, r: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
    a = np.random.uniform(*r)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
    return np.clip(img.astype(np.float32) * a + g * (1 - a), 0, 255).astype(np.uint8)


def random_saturation(img: np.ndarray, r: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= np.random.uniform(*r)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_hue(img: np.ndarray, delta: int = 10) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] += np.random.randint(-delta, delta)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_gaussian_blur(img: np.ndarray, p: float = 0.3, k: int = 5) -> np.ndarray:
    if np.random.rand() < p:
        img = cv2.GaussianBlur(img, (max(3, k if k % 2 else k + 1),) * 2, 0)
    return img


def random_motion_blur(img: np.ndarray, p: float = 0.15) -> np.ndarray:
    if np.random.rand() < p:
        k = int(np.random.choice([5, 7, 9]))
        kn = np.zeros((k, k), dtype=np.float32)
        a = np.random.randint(0, 180)
        dx, dy = int(round(np.cos(np.radians(a)) * (k - 1))), int(round(np.sin(np.radians(a)) * (k - 1)))
        cv2.line(kn, ((k - 1) // 2,) * 2, ((k - 1) // 2 + dx, (k - 1) // 2 + dy), 1, 1)
        kn /= kn.sum()
        img = cv2.filter2D(img, -1, kn)
    return img


def random_jpeg(img: np.ndarray, p: float = 0.4) -> np.ndarray:
    if np.random.rand() < p:
        _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, np.random.randint(30, 100)])
        img = cv2.imdecode(enc, 1)
    return img


def random_noise(img: np.ndarray, p: float = 0.2) -> np.ndarray:
    if np.random.rand() < p:
        n = np.random.randn(*img.shape).astype(np.float32) * np.random.randint(5, 25)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return img


def random_rotation(img: np.ndarray, p: float = 0.3, deg: float = 15) -> np.ndarray:
    if np.random.rand() < p:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), np.random.uniform(-deg, deg), 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
    return img


def random_cutout(img: np.ndarray, p: float = 0.15) -> np.ndarray:
    if np.random.rand() < p:
        h, w = img.shape[:2]
        for _ in range(np.random.randint(1, 3)):
            y = np.random.randint(0, h - 48); x = np.random.randint(0, w - 48)
            img[y:y + np.random.randint(8, 48), x:x + np.random.randint(8, 48)] = np.random.randint(0, 256, (1, 1, 3), dtype=np.uint8)
    return img


def train_augment(face_bgr: np.ndarray) -> np.ndarray:
    """Full augmentation pipeline for training faces."""
    img = face_bgr.copy()
    img = random_hflip(img)
    img = random_rotation(img)
    img = random_brightness(img)
    img = random_contrast(img)
    img = random_saturation(img)
    img = random_hue(img)
    img = random_gaussian_blur(img)
    img = random_motion_blur(img)
    img = random_noise(img)
    img = random_jpeg(img)
    img = random_cutout(img)
    return img


# ═══════════════════════════════════════════════════════════════════
# II. DATASETS
# ═══════════════════════════════════════════════════════════════════

CLASS_NAMES = ["real", "filter", "deepfake"]


class FaceFrameDataset(Dataset):
    """Single-frame face dataset."""

    def __init__(self, root: str, mode: str = "train"):
        self.root, self.mode = root, mode
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        self.samples: List[Tuple[str, int]] = []
        for cls in CLASS_NAMES:
            d = os.path.join(root, cls)
            if not os.path.isdir(d): continue
            lab = self.class_to_idx[cls]
            for vid in sorted(os.listdir(d)):
                vd = os.path.join(d, vid)
                if not os.path.isdir(vd): continue
                for fn in sorted(os.listdir(vd)):
                    if fn.endswith((".npy", ".jpg", ".png", ".jpeg")):
                        self.samples.append((os.path.join(vd, fn), lab))
        if not self.samples:
            raise RuntimeError(f"No samples in {root}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if path.endswith(".npy"):
            t = np.load(path)
            return torch.from_numpy(t), label
        img = cv2.imread(path)
        if self.mode == "train":
            img = train_augment(img)
        proc = preprocess_infer(img)
        return torch.from_numpy(proc), label


class VideoClipDataset(Dataset):
    """Temporal clip dataset (T, C, H, W)."""

    def __init__(self, root: str, seq_len: int = 16, stride: int = 4, mode: str = "train"):
        self.root, self.seq_len, self.stride, self.mode = root, seq_len, stride, mode
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        self.clips: List[Tuple[str, int, int]] = []
        for cls in CLASS_NAMES:
            d = os.path.join(root, cls)
            if not os.path.isdir(d): continue
            lab = self.class_to_idx[cls]
            for vid in sorted(os.listdir(d)):
                vd = os.path.join(d, vid)
                if not os.path.isdir(vd): continue
                fs = sorted(f for f in os.listdir(vd) if f.endswith((".npy", ".jpg", ".png")))
                if len(fs) >= seq_len * stride:
                    self.clips.append((vd, lab, len(fs)))

    def __len__(self): return len(self.clips)

    def _load(self, vd: str, idx: int):
        fs = sorted(f for f in os.listdir(vd) if f.endswith((".npy", ".jpg", ".png")))
        fp = os.path.join(vd, fs[min(idx, len(fs) - 1)])
        if fp.endswith(".npy"):
            t = np.load(fp)
            return torch.from_numpy(t if t.ndim == 3 and t.shape[0] in (1, 3) else np.transpose(t, (2, 0, 1)) / 255.0)
        return torch.from_numpy(preprocess_infer(cv2.imread(fp)))

    def __getitem__(self, idx):
        vd, lab, n = self.clips[idx]
        mx = max(0, n - self.seq_len * self.stride)
        st = random.randint(0, mx) if self.mode == "train" and mx > 0 else 0
        inds = [min(st + i * self.stride, n - 1) for i in range(self.seq_len)]
        clip = torch.stack([self._load(vd, i) for i in inds])
        return clip, lab


# ═══════════════════════════════════════════════════════════════════
# III. FACE DETECTION — MediaPipe
# ═══════════════════════════════════════════════════════════════════

class FaceDetector:
    """MediaPipe-based face detector with padding."""

    def __init__(self, conf: float = 0.5):
        self.det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=conf)

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Returns list of (face_crop, (x1, y1, x2, y2))."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.det.process(rgb)
        faces = []
        if res.detections:
            for d in res.detections:
                b = d.location_data.relative_bounding_box
                x1, y1 = int(b.xmin * w), int(b.ymin * h)
                x2, y2 = int((b.xmin + b.width) * w), int((b.ymin + b.height) * h)
                pw, ph = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
                x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
                x2, y2 = min(w, x2 + pw), min(h, y2 + ph)
                face = frame_bgr[y1:y2, x1:x2]
                if face.size > 0:
                    faces.append((face, (x1, y1, x2, y2)))
        return faces


# ═══════════════════════════════════════════════════════════════════
# IV. VISUALISATION
# ═══════════════════════════════════════════════════════════════════

COLOURS = {"real": (76, 205, 76), "filter": (255, 165, 0), "deepfake": (66, 66, 245)}
EMOJIS  = {"real": "✅", "filter": "🎭", "deepfake": "⚠️"}
FONT = cv2.FONT_HERSHEY_DUPLEX


def draw_overlay(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                 cls_name: str, confidence: float, fps: Optional[float] = None) -> np.ndarray:
    """Draw bounding box, label, confidence bar, and FPS."""
    x1, y1, x2, y2 = bbox
    colour = COLOURS.get(cls_name, (200, 200, 200))

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    label = f"{EMOJIS.get(cls_name, '?')} {cls_name.upper()}"
    (lw, lh), _ = cv2.getTextSize(label, FONT, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - lh - 14), (x1 + lw + 20, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 10, y1 - 8), FONT, 0.7, (255, 255, 255), 2)

    bw = x2 - x1
    cv2.rectangle(frame, (x1, y2 + 10), (x1 + bw, y2 + 18), (50, 50, 50), -1)
    cv2.rectangle(frame, (x1, y2 + 10), (x1 + int(bw * confidence), y2 + 18), colour, -1)
    cv2.putText(frame, f"{confidence:.0%}", (x1 + 4, y2 + 17), FONT, 0.4, (255, 255, 255), 1)

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, 0.7, (0, 255, 0), 2)

    return frame


def draw_panel(frame: np.ndarray, probs: np.ndarray, pos: Tuple[int, int] = (10, 80)) -> np.ndarray:
    """Side panel with per-class confidence bars."""
    px, py = pos
    n = len(probs)
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + 200, py + n * 25 + 20), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "PREDICTIONS", (px + 10, py + 18), FONT, 0.5, (200, 200, 200), 1)
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        y = py + 30 + i * 25
        c = COLOURS.get(name, (200, 200, 200))
        cv2.putText(frame, f"{name:<12}", (px + 10, y + 14), FONT, 0.45, c, 1)
        bx, bw = px + 110, 80
        cv2.rectangle(frame, (bx, y + 2), (bx + bw, y + 18), (40, 40, 40), -1)
        cv2.rectangle(frame, (bx, y + 2), (bx + int(bw * prob), y + 18), c, -1)
        cv2.putText(frame, f"{prob:.1%}", (bx + int(bw * prob) + 4, y + 14), FONT, 0.35, (200, 200, 200), 1)
    return frame


# ═══════════════════════════════════════════════════════════════════
# V. TEMPORAL BUFFER (smoothing)
# ═══════════════════════════════════════════════════════════════════

class TemporalBuffer:
    """Sliding window average for frame-level predictions."""
    def __init__(self, window: int = 5):
        self.buf: Deque[np.ndarray] = deque(maxlen=window)
    def update(self, prob: np.ndarray) -> np.ndarray:
        self.buf.append(prob)
        return np.mean(list(self.buf), axis=0)
    def reset(self):
        self.buf.clear()


# ═══════════════════════════════════════════════════════════════════
# VI. INFERENCE — Core
# ═══════════════════════════════════════════════════════════════════

def infer_frame(model: nn.Module, face_bgr: np.ndarray, device: str) -> np.ndarray:
    """Run model on a single face crop."""
    tensor = torch.from_numpy(preprocess_infer(face_bgr)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    return torch.softmax(out, dim=1).cpu().numpy()[0]


# ═══════════════════════════════════════════════════════════════════
# VII. INFERENCE — Modes
# ═══════════════════════════════════════════════════════════════════

def run_webcam(model: nn.Module, device: str, cam_id: int = 0):
    """Run real-time webcam inference."""
    det = FaceDetector()
    buf = TemporalBuffer()
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"Webcam {cam_id} — press 'q' quit, 'r' reset buffer")
    prev = time.perf_counter()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        now = time.perf_counter()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        faces = det.detect(frame)
        if faces:
            for face, bbox in faces:
                prob = infer_frame(model, cv2.resize(face, (224, 224)), device)
                smoothed = buf.update(prob)
                cls = CLASS_NAMES[smoothed.argmax()]
                frame = draw_overlay(frame, bbox, cls, smoothed.max(), fps)
                frame = draw_panel(frame, smoothed)
        else:
            frame = draw_panel(frame, np.zeros(3))
            cv2.putText(frame, "No face", (10, 60), FONT, 0.6, (100, 100, 100), 1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, 0.7, (0, 255, 0), 2)
        cv2.imshow("Deepfake Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): buf.reset(); print("Buffer reset")
    cap.release(); cv2.destroyAllWindows()


def run_video(model: nn.Module, device: str, video_path: str, output_path: Optional[str] = None):
    """Run inference on a video file."""
    det = FaceDetector()
    buf = TemporalBuffer()
    cap = cv2.VideoCapture(video_path)
    w, h, fps_in = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
    prev = time.perf_counter()
    pbar = tqdm(total=total, desc="Processing", ncols=80, file=sys.stdout)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        now = time.perf_counter()
        fps = 1.0 / max(now - prev, 1e-6); prev = now
        for face, bbox in det.detect(frame):
            prob = infer_frame(model, cv2.resize(face, (224, 224)), device)
            sm = buf.update(prob)
            frame = draw_overlay(frame, bbox, CLASS_NAMES[sm.argmax()], sm.max(), fps)
            frame = draw_panel(frame, sm)
        if writer: writer.write(frame)
        pbar.update(1)
    pbar.close(); cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    if output_path: print(f"Saved: {output_path}")


def run_image(model: nn.Module, device: str, image_path: str, output_path: Optional[str] = None):
    """Run inference on a single image."""
    frame = cv2.imread(image_path)
    if frame is None: print(f"Cannot read {image_path}"); return
    det = FaceDetector()
    faces = det.detect(frame)
    if not faces:
        print("No faces detected.")
    for face, bbox in faces:
        prob = infer_frame(model, cv2.resize(face, (224, 224)), device)
        frame = draw_overlay(frame, bbox, CLASS_NAMES[prob.argmax()], prob.max())
        frame = draw_panel(frame, prob)
    if output_path:
        cv2.imwrite(output_path, frame); print(f"Saved: {output_path}")
    else:
        cv2.imshow("Deepfake Detection", frame)
        print("Press any key."); cv2.waitKey(0)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════
# VIII. MODEL LOADING & EXPORT
# ═══════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, temporal: bool = False, device: str = "cuda") -> nn.Module:
    if temporal:
        model = TemporalModel(num_classes=3)
    else:
        model = EfficientNetClassifier(num_classes=3)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    # Validate parameter keys and warn on mismatch
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    matched = model_keys & ckpt_keys
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    if missing:
        print(f"  [WARN] Missing {len(missing)} key(s) in checkpoint: {sorted(missing)[:5]}...")
    if unexpected:
        print(f"  [WARN] {len(unexpected)} unexpected key(s) in checkpoint: {sorted(unexpected)[:5]}...")
    print(f"  Loaded {len(matched)}/{len(model_keys)} parameter groups")
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def export_onnx(model: nn.Module, output_path: str, temporal: bool = False,
                opset: int = 17, dynamic_batch: bool = True):
    """Export to ONNX format."""
    device = next(model.parameters()).device
    if temporal:
        dummy = torch.randn(1, 16, 3, 224, 224, device=device)
    else:
        dummy = torch.randn(1, 3, 224, 224, device=device)
    dynamic = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None
    torch.onnx.export(model, dummy, output_path,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes=dynamic, opset_version=opset,
                      do_constant_folding=True)
    print(f"ONNX exported: {output_path} | Input: {list(dummy.shape)}")
    # Quick verify
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider" if device == "cpu"
                                   else "CUDAExecutionProvider"])
        out = sess.run(None, {"input": dummy.cpu().numpy().astype(np.float32)})
        print(f"  Verify OK — output shape: {list(out[0].shape)}")
    except ImportError:
        print("  (install onnxruntime to verify)")


def export_torchscript(model: nn.Module, output_path: str, temporal: bool = False):
    """Export as TorchScript (traced)."""
    device = next(model.parameters()).device
    if temporal:
        dummy = torch.randn(1, 16, 3, 224, 224, device=device)
    else:
        dummy = torch.randn(1, 3, 224, 224, device=device)
    traced = torch.jit.trace(model, dummy)
    traced.save(output_path)
    print(f"TorchScript exported: {output_path} | Input: {list(dummy.shape)}")
    # Quick verify
    loaded = torch.jit.load(output_path)
    out = loaded(dummy)
    print(f"  Verify OK — output shape: {list(out.shape)}")


# ═══════════════════════════════════════════════════════════════════
# IX. ENSEMBLE
# ═══════════════════════════════════════════════════════════════════

def ensemble_predict(frame_model: nn.Module, temporal_model: nn.Module,
                     face_bgr: np.ndarray, device: str,
                     weights: Tuple[float, float] = (0.4, 0.6)) -> np.ndarray:
    """Weighted soft voting from frame + temporal models."""
    tensor = torch.from_numpy(preprocess_infer(cv2.resize(face_bgr, (224, 224)))).unsqueeze(0).to(device)
    with torch.no_grad():
        f_prob = torch.softmax(frame_model(tensor), dim=1)
        # Temporal model expects (1, T, C, H, W) — repeat single frame T times
        t_tensor = tensor.unsqueeze(1).repeat(1, 16, 1, 1, 1)
        t_prob = torch.softmax(temporal_model(t_tensor), dim=1)
    return (weights[0] * f_prob + weights[1] * t_prob).cpu().numpy()[0]
