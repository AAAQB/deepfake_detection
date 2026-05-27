import os
import sys
import time
import random
import argparse
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from preprocessing import (
    extract_faces_from_image,  
    extract_frames_from_video, 
    preprocess_train,          
    preprocess_infer,           
    random_horizontal_flip,     
    random_brightness,
    random_compression,
)

from EfficientNet import EfficientNetClassifier
from LSTM import TemporalModel


# ═══════════════════════════════════════════════════════════════════
# I. DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════

def train_augment(face_bgr: np.ndarray) -> np.ndarray:

    img = face_bgr.copy()
    img = random_horizontal_flip(img)  
    img = random_brightness(img)        
    img = random_compression(img)       
    return img


# ═══════════════════════════════════════════════════════════════════
# II. DATASETS
# ═══════════════════════════════════════════════════════════════════

CLASS_NAMES = ["real", "filter", "deepfake"]


class FaceFrameDataset(Dataset):

    def __init__(self, root: str, mode: str = "train"):
        self.root, self.mode = root, mode
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        self.samples: List[Tuple[str, int]] = []
        for cls in CLASS_NAMES:
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            lab = self.class_to_idx[cls]
            for vid in sorted(os.listdir(d)):
                vd = os.path.join(d, vid)
                if not os.path.isdir(vd):
                    continue
                for fn in sorted(os.listdir(vd)):
                    if fn.endswith((".npy", ".jpg", ".png", ".jpeg")):
                        self.samples.append((os.path.join(vd, fn), lab))
        if not self.samples:
            raise RuntimeError(f"No samples found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        if path.endswith(".npy"):
            t = np.load(path)
            return torch.from_numpy(t), label

        img = cv2.imread(path)
        if self.mode == "train":
            proc = preprocess_train(img)
        else:
            proc = preprocess_infer(img)

        return torch.from_numpy(proc), label


class VideoClipDataset(Dataset):

    def __init__(self, root: str, seq_len: int = 16, stride: int = 4, mode: str = "train"):
        self.root, self.seq_len, self.stride, self.mode = root, seq_len, stride, mode
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        self.clips: List[Tuple[str, int, int]] = []
        for cls in CLASS_NAMES:
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            lab = self.class_to_idx[cls]
            for vid in sorted(os.listdir(d)):
                vd = os.path.join(d, vid)
                if not os.path.isdir(vd):
                    continue
                fs = sorted(f for f in os.listdir(vd) if f.endswith((".npy", ".jpg", ".png")))
                if len(fs) >= seq_len * stride:
                    self.clips.append((vd, lab, len(fs)))

    def __len__(self):
        return len(self.clips)

    def _load(self, vd: str, idx: int) -> torch.Tensor:

        fs = sorted(f for f in os.listdir(vd) if f.endswith((".npy", ".jpg", ".png")))
        fp = os.path.join(vd, fs[min(idx, len(fs) - 1)])

        if fp.endswith(".npy"):
            t = np.load(fp)
            if t.ndim == 3 and t.shape[0] in (1, 3):
                return torch.from_numpy(t.astype(np.float32))
            
            return torch.from_numpy(np.transpose(t, (2, 0, 1)).astype(np.float32) / 255.0)

        img = cv2.imread(fp)
        proc = preprocess_infer(img)   # → CHW float32, ImageNet normalized
        return torch.from_numpy(proc)

    def __getitem__(self, idx):
        vd, lab, n = self.clips[idx]
        mx = max(0, n - self.seq_len * self.stride)
        st = random.randint(0, mx) if self.mode == "train" and mx > 0 else 0
        inds = [min(st + i * self.stride, n - 1) for i in range(self.seq_len)]
        clip = torch.stack([self._load(vd, i) for i in inds])  # (T, C, H, W)
        return clip, lab


# ═══════════════════════════════════════════════════════════════════
# III. FACE DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_faces_with_bbox(
    frame_bgr: np.ndarray,
    pad: float = 0.15
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    import mediapipe as mp  

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    from preprocessing import mp_face as _mp_face
    results = _mp_face.process(rgb)

    faces_with_bbox = []
    if results.detections:
        for det in results.detections:
            b = det.location_data.relative_bounding_box
            x1 = int(b.xmin * w)
            y1 = int(b.ymin * h)
            x2 = int((b.xmin + b.width) * w)
            y2 = int((b.ymin + b.height) * h)
            
            pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
            x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
            x2, y2 = min(w, x2 + pw), min(h, y2 + ph)
            face = frame_bgr[y1:y2, x1:x2]
            if face.size > 0:
                faces_with_bbox.append((face, (x1, y1, x2, y2)))

    return faces_with_bbox


# ═══════════════════════════════════════════════════════════════════
# IV. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

COLOURS = {"real": (76, 205, 76), "filter": (255, 165, 0), "deepfake": (66, 66, 245)}
EMOJIS  = {"real": "✅", "filter": "🎭", "deepfake": "⚠️"}
FONT = cv2.FONT_HERSHEY_DUPLEX


def draw_overlay(frame, bbox, cls_name, confidence, fps=None):
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


def draw_panel(frame, probs, pos=(10, 80)):
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
# V. TEMPORAL BUFFE
# ═══════════════════════════════════════════════════════════════════

class TemporalBuffer:
    
    def __init__(self, window: int = 5):
        self.buf: Deque[np.ndarray] = deque(maxlen=window)

    def update(self, prob: np.ndarray) -> np.ndarray:
        self.buf.append(prob)
        return np.mean(list(self.buf), axis=0)

    def reset(self):
        self.buf.clear()


# ═══════════════════════════════════════════════════════════════════
# VI. INTERFACE
# ═══════════════════════════════════════════════════════════════════

def infer_frame(model: nn.Module, face_bgr: np.ndarray, device: str) -> np.ndarray:
    tensor = torch.from_numpy(preprocess_infer(face_bgr)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    return torch.softmax(out, dim=1).cpu().numpy()[0]

def run_webcam(model: nn.Module, device: str, cam_id: int = 0):
 
    buf = TemporalBuffer()
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"Webcam {cam_id} — press 'q' quit, 'r' reset buffer")
    prev = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        now = time.perf_counter()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        faces = detect_faces_with_bbox(frame)
        if faces:
            for face, bbox in faces:
                prob = infer_frame(model, face, device)
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
        if key == ord('q'):
            break
        if key == ord('r'):
            buf.reset()
            print("Buffer reset")

    cap.release()
    cv2.destroyAllWindows()


def run_video(model: nn.Module, device: str, video_path: str, output_path: Optional[str] = None):

    buf = TemporalBuffer()
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(3))
    h = int(cap.get(4))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_in, (w, h)
        )

    prev = time.perf_counter()
    pbar = tqdm(total=total, desc="Processing", ncols=80, file=sys.stdout)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        now = time.perf_counter()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        for face, bbox in detect_faces_with_bbox(frame):
            prob = infer_frame(model, face, device)
            sm = buf.update(prob)
            frame = draw_overlay(frame, bbox, CLASS_NAMES[sm.argmax()], sm.max(), fps)
            frame = draw_panel(frame, sm)

        if writer:
            writer.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    if output_path:
        print(f"Saved: {output_path}")


def run_image(model: nn.Module, device: str, image_path: str, output_path: Optional[str] = None):

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Cannot read {image_path}")
        return

    faces = detect_faces_with_bbox(frame)
    if not faces:
        print("No faces detected.")

    for face, bbox in faces:
        prob = infer_frame(model, face, device)
        frame = draw_overlay(frame, bbox, CLASS_NAMES[prob.argmax()], prob.max())
        frame = draw_panel(frame, prob)

    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"Saved: {output_path}")
    else:
        cv2.imshow("Deepfake Detection", frame)
        print("Press any key.")
        cv2.waitKey(0)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════
# VIII. LOAD MODEL
# ═══════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, temporal: bool = False, device: str = "cuda") -> nn.Module:
    """
    从 checkpoint 文件加载模型权重。
    checkpoint 是训练过程中保存的模型状态字典（见 train.py）。
    """
    if temporal:
        model = TemporalModel(num_classes=3)
    else:
        model = EfficientNetClassifier(num_classes=3)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.removeprefix("module."): v for k, v in state.items()}

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    matched = model_keys & ckpt_keys
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        print(f"  [WARN] Missing {len(missing)} key(s): {sorted(missing)[:5]}...")
    if unexpected:
        print(f"  [WARN] {len(unexpected)} unexpected key(s): {sorted(unexpected)[:5]}...")
    print(f"  Loaded {len(matched)}/{len(model_keys)} parameter groups")

    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def export_onnx(model: nn.Module, output_path: str, temporal: bool = False,
                opset: int = 17, dynamic_batch: bool = True):
    device = next(model.parameters()).device
    dummy = torch.randn(1, 16, 3, 224, 224, device=device) if temporal else torch.randn(1, 3, 224, 224, device=device)
    dynamic = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None
    torch.onnx.export(model, dummy, output_path,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes=dynamic, opset_version=opset,
                      do_constant_folding=True)
    print(f"ONNX exported: {output_path}")


def export_torchscript(model: nn.Module, output_path: str, temporal: bool = False):
    device = next(model.parameters()).device
    dummy = torch.randn(1, 16, 3, 224, 224, device=device) if temporal else torch.randn(1, 3, 224, 224, device=device)
    traced = torch.jit.trace(model, dummy)
    traced.save(output_path)
    print(f"TorchScript exported: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# IX.MODEL ENSEMBLE
# ═══════════════════════════════════════════════════════════════════

def ensemble_predict(
    frame_model: nn.Module,
    temporal_model: nn.Module,
    face_bgr: np.ndarray,
    device: str,
    weights: Tuple[float, float] = (0.4, 0.6)
) -> np.ndarray:

    proc = preprocess_infer(face_bgr)   # CHW float32, ImageNet normalized
    tensor = torch.from_numpy(proc).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        f_prob = torch.softmax(frame_model(tensor), dim=1)

        t_tensor = tensor.unsqueeze(1).repeat(1, 16, 1, 1, 1)
        t_prob = torch.softmax(temporal_model(t_tensor), dim=1)

    fused = weights[0] * f_prob + weights[1] * t_prob
    return fused.cpu().numpy()[0]
