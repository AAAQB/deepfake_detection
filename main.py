#!/usr/bin/env python
"""
main.py — Deepfake Detection CLI Entry Point

Usage:
  python main.py preprocess                         # Extract faces from raw videos
  python main.py train                              # Train frame-level model
  python main.py train --temporal                   # Train temporal model
  python main.py webcam <checkpoint.pt>             # Real-time webcam inference
  python main.py video <ckpt.pt> <video.mp4>        # Video file inference
  python main.py image <ckpt.pt> <photo.jpg>        # Single image inference
  python main.py export <ckpt.pt>                   # Export to ONNX + TorchScript
  python main.py ensemble <frame.pt> <temporal.pt>  # Ensemble inference
"""

import sys
import argparse
import cv2

from preprocessing import process_videos
from train import CFG as TrainCFG, train as train_model

# ─── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Detection — Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("command", type=str,
                        choices=["preprocess", "train", "train_temporal",
                                 "webcam", "video", "image", "export", "ensemble"],
                        help="Pipeline step to run")
    parser.add_argument("args", nargs="*", help="Additional positional arguments")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (cuda / cpu)")

    parsed, _ = parser.parse_known_args()
    argv = sys.argv[2:]  # Forward remaining args

    cmd = parsed.command

    if cmd == "preprocess":
        process_videos()

    elif cmd == "train":
        if parsed.epochs: TrainCFG.max_epochs = parsed.epochs
        if parsed.batch_size: TrainCFG.batch_size = parsed.batch_size
        if parsed.lr: TrainCFG.learning_rate = parsed.lr
        train_model(temporal=False, resume=parsed.resume)

    elif cmd == "train_temporal":
        if parsed.epochs: TrainCFG.max_epochs = parsed.epochs
        if parsed.batch_size: TrainCFG.batch_size = parsed.batch_size
        if parsed.lr: TrainCFG.learning_rate = parsed.lr
        train_model(temporal=True, resume=parsed.resume)

    elif cmd in ("webcam", "video", "image", "export", "ensemble"):
        # These route to interface.py
        _run_interface_command(cmd, sys.argv[2:], parsed.device)

    else:
        parser.print_help()


def _run_interface_command(cmd: str, argv: list, device: str):
    import interface as iface

    if cmd == "webcam":
        if len(argv) < 1:
            print("Usage: python main.py webcam <checkpoint.pt> [camera_id]"); return
        ckpt = argv[0]
        cam = int(argv[1]) if len(argv) > 1 else 0
        model = iface.load_model(ckpt, temporal=False, device=device)
        iface.run_webcam(model, device, cam)

    elif cmd == "video":
        if len(argv) < 2:
            print("Usage: python main.py video <checkpoint.pt> <video.mp4> [output.mp4]"); return
        ckpt, vid_path = argv[0], argv[1]
        out = argv[2] if len(argv) > 2 else "output_detection.mp4"
        model = iface.load_model(ckpt, temporal=False, device=device)
        iface.run_video(model, device, vid_path, out)

    elif cmd == "image":
        if len(argv) < 2:
            print("Usage: python main.py image <checkpoint.pt> <image.jpg> [output.jpg]"); return
        ckpt, img_path = argv[0], argv[1]
        out = argv[2] if len(argv) > 2 else None
        model = iface.load_model(ckpt, temporal=False, device=device)
        iface.run_image(model, device, img_path, out)

    elif cmd == "export":
        if len(argv) < 1:
            print("Usage: python main.py export <checkpoint.pt> [--dir exported_models]"); return
        ckpt = argv[0]
        outdir = "exported_models"
        if "--dir" in argv:
            outdir = argv[argv.index("--dir") + 1]
        import os; os.makedirs(outdir, exist_ok=True)
        base = os.path.splitext(os.path.basename(ckpt))[0]

        for temporal in [False, True]:
            try:
                model = iface.load_model(ckpt, temporal=temporal, device=device)
                iface.export_onnx(model, os.path.join(outdir, f"{base}_{'temporal' if temporal else 'frame'}.onnx"),
                                  temporal=temporal)
                iface.export_torchscript(model, os.path.join(outdir, f"{base}_{'temporal' if temporal else 'frame'}.pt"),
                                         temporal=temporal)
            except RuntimeError:
                continue  # Try next model type

    elif cmd == "ensemble":
        if len(argv) < 2:
            print("Usage: python main.py ensemble <frame.pt> <temporal.pt> [--image img.jpg]"); return
        f_ckpt, t_ckpt = argv[0], argv[1]
        f_model = iface.load_model(f_ckpt, temporal=False, device=device)
        t_model = iface.load_model(t_ckpt, temporal=True, device=device)
        print("Models loaded. Use --image flag to test.")
        if "--image" in argv:
            img_path = argv[argv.index("--image") + 1]
            frame = cv2.imread(img_path)
            det = iface.FaceDetector()
            for face, bbox in det.detect(frame):
                prob = iface.ensemble_predict(f_model, t_model, cv2.resize(face, (224, 224)), device)
                frame = iface.draw_overlay(frame, bbox, iface.CLASS_NAMES[prob.argmax()], prob.max())
                frame = iface.draw_panel(frame, prob)
            cv2.imshow("Ensemble", frame)
            cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
