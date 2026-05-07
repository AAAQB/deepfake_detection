

import io
import os
import glob
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

CLASS_NAMES = ["real", "filter", "deepfake"]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FaceFrameDataset(Dataset):
    """
    Loads pre-extracted face crops (.npy) organized as:
        dataset_faces/{class_name}/{video_id}/*.npy

    Modes:
        seq_len=None  → single frame (EfficientNet, PDF Section 8)
        seq_len=T     → sequence of T frames (CNN+LSTM, PDF Section 10)
    """

    def __init__(self, root_dir, seq_len=None, augment=True):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.augment = augment
        self.samples = []

        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            label = CLASS_MAP[cls_name]

            for video_dir in sorted(glob.glob(os.path.join(cls_dir, "*"))):
                if not os.path.isdir(video_dir):
                    continue

                npy_files = sorted(glob.glob(os.path.join(video_dir, "*.npy")))

                if seq_len is None:
                    # Single frame mode: each .npy is one sample
                    for npy_path in npy_files:
                        self.samples.append((npy_path, label))
                else:
                    # Sequence mode: group into chunks of seq_len
                    npy_paths_list = npy_files
                    for i in range(0, len(npy_paths_list), seq_len):
                        chunk = npy_paths_list[i:i + seq_len]
                        if len(chunk) == seq_len:
                            self.samples.append((chunk, label))

    def __len__(self):
        return len(self.samples)

    def _augment(self, tensor):
        """PDF Section 7 augmentation: flip, rotation, brightness, blur, compression, color jitter."""
        t = tensor.clone()
        c, h, w = t.shape

        # Random horizontal flip (p=0.5) — PDF §7
        if random.random() < 0.5:
            t = torch.flip(t, dims=[-1])

        # Random rotation (±10°) — PDF §7
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            # Use affine grid for proper rotation with bilinear interpolation
            theta = torch.tensor([[
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                [np.sin(np.radians(angle)),  np.cos(np.radians(angle)), 0]
            ]], dtype=torch.float32)
            grid = F.affine_grid(theta, (1, c, h, w), align_corners=False)
            t = F.grid_sample(t.unsqueeze(0), grid, align_corners=False).squeeze(0)

        # Random zoom/crop (p=0.3)
        if random.random() < 0.3:
            crop_h, crop_w = int(h * random.uniform(0.85, 1.0)), int(w * random.uniform(0.85, 1.0))
            y = random.randint(0, h - crop_h)
            x = random.randint(0, w - crop_w)
            cropped = t[:, y:y + crop_h, x:x + crop_w]
            t = F.interpolate(cropped.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)

        # Brightness variation (p=0.4) — PDF §7
        if random.random() < 0.4:
            brightness = random.uniform(0.7, 1.3)
            t = t * brightness

        # Gaussian blur (p=0.25) — PDF §7
        if random.random() < 0.25:
            ks = random.choice([3, 5])
            sigma = 0.3 * ((ks - 1) * 0.5 - 1) + 0.8
            ax = torch.arange(ks, device=t.device).float() - ks // 2
            gauss = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
            kernel_1d = gauss / gauss.sum()
            kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
            blur_kernel = kernel_2d.expand(c, 1, ks, ks).to(t.device, t.dtype)
            t = t.unsqueeze(0)
            t = F.conv2d(t, blur_kernel, padding=ks // 2, groups=c)
            t = t.squeeze(0)

        # JPEG compression simulation (p=0.2) — PDF §7
        # Uses real JPEG encode/decode via PIL for authentic block artifacts
        if random.random() < 0.2:
            quality = random.randint(50, 90)
            t_np = (t.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(t_np)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, subsampling=0)
            buf.seek(0)
            decoded = Image.open(buf)
            t_decoded = np.array(decoded, dtype=np.float32).transpose(2, 0, 1) / 255.0
            t = torch.from_numpy(t_decoded).to(tensor.dtype)

        # Color jitter (p=0.3)
        if random.random() < 0.3:
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            mean = t.mean(dim=(1, 2), keepdim=True)
            t = (t - mean) * contrast + mean
            gray = t.mean(dim=0, keepdim=True)
            t = t * saturation + gray * (1 - saturation)

        t = torch.clamp(t, -3.0, 3.0)
        return t

    def __getitem__(self, idx):
        item = self.samples[idx]

        if isinstance(item[0], list):
            # Sequence mode
            npy_paths, label = item
            frames = []
            for p in npy_paths:
                arr = np.load(p)
                t = torch.from_numpy(arr)
                if self.augment:
                    t = self._augment(t)
                frames.append(t)
            tensor = torch.stack(frames, dim=0)  # (T, 3, 224, 224)
        else:
            # Single frame mode
            npy_path, label = item
            arr = np.load(npy_path)
            tensor = torch.from_numpy(arr)  # (3, 224, 224)
            if self.augment:
                tensor = self._augment(tensor)

        return tensor, label


def create_dataloaders(root_dir, batch_size, val_split=0.15,
                       seq_len=None, augment_train=True, num_workers=4):
    """Convenience function to create train/val loaders."""
    full_dataset = FaceFrameDataset(root_dir, seq_len=seq_len, augment=augment_train)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test
    ds = FaceFrameDataset("dataset_faces", seq_len=None, augment=False)
    print(f"Single-frame dataset: {len(ds)} samples")
    if len(ds) > 0:
        x, y = ds[0]
        print(f"  Sample: {x.shape} → label {y} ({CLASS_NAMES[y]})")

    ds_seq = FaceFrameDataset("dataset_faces", seq_len=8, augment=False)
    print(f"Sequence dataset (T=8): {len(ds_seq)} samples")
    if len(ds_seq) > 0:
        x, y = ds_seq[0]
        print(f"  Sample: {x.shape} → label {y} ({CLASS_NAMES[y]})")
