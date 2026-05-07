

import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import EfficientNet as EN
import LSTM as LS
from dataset import FaceFrameDataset, CLASS_NAMES


def train_epoch(model, loader, criterion, optimizer, device, amp, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc="  Train", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        B = inputs.size(0)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += loss.item() * B
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += B

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, amp):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        B = inputs.size(0)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * B
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += B

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="DeepGuard Training (PDF Spec)")
    parser.add_argument("--model_type", type=str, default="efficientnet",
                        choices=["efficientnet", "temporal"],
                        help="Model architecture (PDF Sections 8/10)")
    parser.add_argument("--data_root", type=str, default="dataset_faces")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5,
                        help="Minimum LR for cosine annealing")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout rate for classification head")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="LSTM hidden size (temporal model only)")
    parser.add_argument("--seq_len", type=int, default=8,
                        help="Sequence length for temporal model")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze pre-trained CNN backbone")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available() and not args.no_amp
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    print("=" * 60)
    print("DeepGuard Training (PDF Specification)")
    print(f"  Model:      {'CNN+LSTM' if args.model_type == 'temporal' else 'EfficientNet-B4'}")
    print(f"  Device:     {device}  |  AMP: {amp}")
    print(f"  Batch:      {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Dropout:    {args.dropout}")
    print(f"  Workers:    {args.num_workers}")
    if args.freeze_backbone:
        print(f"  Backbone:   frozen (only training classification head)")
    if args.model_type == "temporal":
        print(f"  Seq Len:    {args.seq_len}")
        print(f"  LSTM Dim:   {args.hidden_size}")
    print("=" * 60)

    # Model
    seq_len = args.seq_len if args.model_type == "temporal" else None
    if args.model_type == "efficientnet":
        model = EN.EfficientNet(
            num_classes=3,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
            pretrained=True,
        )
    else:
        model = LS.CNNLSTM(
            num_classes=3,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
            pretrained=True,
            seq_len=seq_len,
        )

    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters:  {params:,}  (trainable: {trainable:,})")

    # Data
    full_dataset = FaceFrameDataset(args.data_root, seq_len=seq_len, augment=True)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()

    # Separate param groups: higher LR for the new head, lower for backbone (if unfrozen)
    if args.freeze_backbone:
        optimizer = optim.AdamW(model.classifier.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay)

    # Cosine annealing LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min,
    )

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if amp else None

    # Logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.model_type}_{timestamp}"
    writer = SummaryWriter(os.path.join(args.log_dir, exp_name))
    ckpt_dir = os.path.join(args.checkpoint_dir, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(ckpt_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\nTraining for {args.epochs} epochs (patience={args.patience})...")

    for epoch in range(args.epochs):
        t0 = time.perf_counter()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, amp, scaler,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, amp)
        scheduler.step()

        epoch_time = time.perf_counter() - t0

        # Logging
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {epoch_time:.1f}s")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "train_acc": train_acc,
            "args": vars(args),
        }, os.path.join(ckpt_dir, "last.pt"))

        # Best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "train_acc": train_acc,
                "args": vars(args),
            }, os.path.join(ckpt_dir, "best.pt"))
            print(f"  ★ New best: {val_acc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"  Early stopping triggered (no improvement for {args.patience} epochs)")
            break

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {ckpt_dir}/best.pt")

    # ================================================================
    # Model export (ONNX) — PDF Section 12: Deployment
    # ================================================================
    try:
        model.eval()
        if args.model_type == "efficientnet":
            dummy = torch.randn(1, 3, 224, 224).to(device)
            onnx_path = os.path.join(ckpt_dir, "model.onnx")
        else:
            dummy = torch.randn(1, seq_len, 3, 224, 224).to(device)
            onnx_path = os.path.join(ckpt_dir, "model_temporal.onnx")

        with torch.no_grad():
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=17,
            )
        print(f"Exported ONNX: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"ONNX export skipped: {e}")

    return best_val_acc


if __name__ == "__main__":
    main()
