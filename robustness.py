import os
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -------------------------
# Image corruption functions
# -------------------------

def add_gaussian_noise(img, sigma=20):
    noise = np.random.normal(0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_blur(img, k=5):
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def adjust_brightness(img, factor=1.3):
    img = img.astype(np.float32)
    img *= factor
    return np.clip(img, 0, 255).astype(np.uint8)


def adjust_contrast(img, factor=1.3):
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    img = (img - mean) * factor + mean
    return np.clip(img, 0, 255).astype(np.uint8)


def rotate(img, angle=10):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def jpeg_compress(img, quality=50):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    dec = cv2.imdecode(enc, 1)
    return dec

import random
import glob

# -------------------------
# Dataset loader (for .npy or .jpg)
# -------------------------

class FaceDataset:
    def __init__(self, root_dir):
        self.samples = []
        self.labels = {"real": 0, "filter": 1, "deepfake": 2}

        for cls in ["real", "filter", "deepfake"]:
            paths = glob.glob(os.path.join(root_dir, cls, "**/*.*"), recursive=True)

            for p in paths:
                if p.endswith(".npy") or p.endswith(".jpg") or p.endswith(".png"):
                    self.samples.append((p, self.labels[cls]))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def load_image(self, path):
        if path.endswith(".npy"):
            img = np.load(path)
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            img = (img * 255).astype(np.uint8)
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.load_image(path)
        return img, label


# -------------------------
# Model inference wrapper
# -------------------------

def predict_image(model, img, device, transform=None):
    model.eval()

    if transform is not None:
        img = transform(img)

    if isinstance(img, np.ndarray):
        img = torch.tensor(img).float()

    if img.ndim == 3:
        img = img.unsqueeze(0)

    img = img.to(device)

    with torch.no_grad():
        out = model(img)
        pred = torch.argmax(out, dim=1).cpu().numpy()[0]

    return pred
# -------------------------
# Robustness evaluation core
# -------------------------

def evaluate_robustness(model, dataset, device):
    results = []

    transforms = {
        "original": lambda x: x,
        "noise20": lambda x: add_gaussian_noise(x, 20),
        "noise30": lambda x: add_gaussian_noise(x, 30),
        "blur3": lambda x: add_blur(x, 3),
        "blur5": lambda x: add_blur(x, 5),
        "brightness_1.3": lambda x: adjust_brightness(x, 1.3),
        "brightness_0.7": lambda x: adjust_brightness(x, 0.7),
        "contrast_1.3": lambda x: adjust_contrast(x, 1.3),
        "contrast_0.7": lambda x: adjust_contrast(x, 0.7),
        "rotate_10": lambda x: rotate(x, 10),
        "rotate_-10": lambda x: rotate(x, -10),
        "jpeg_70": lambda x: jpeg_compress(x, 70),
        "jpeg_50": lambda x: jpeg_compress(x, 50),
    }

    for name, func in transforms.items():
        y_true = []
        y_pred = []

        for i in range(len(dataset)):
            img, label = dataset[i]

            try:
                img_aug = func(img)
                pred = predict_image(model, img_aug, device)

                y_true.append(label)
                y_pred.append(pred)

            except Exception as e:
                continue

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        results.append({
            "condition": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        print(f"[{name}] acc={acc:.4f} f1={f1:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("robustness_results.csv", index=False)

    return df

#Usage:dataset = FaceDataset("dataset_faces")
df = evaluate_robustness(model, dataset, device)