#!/usr/bin/env python
"""
main.py — Deepfake Detection CLI Entry Point
"""

import os as _os

# Prevent CPU deadlock in Windows multiprocessing environments
_os.environ["TF_NUM_INTEROP_THREADS"] = "1"
_os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
_os.environ["XNNPACK_FORCE_SINGLE_THREAD"] = "1"

import sys
import argparse
import cv2
import multiprocessing


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

    parsed, unknown = parser.parse_known_args()

    argv = parsed.args
    cmd = parsed.command

    # Lazy imports to minimize heavy dependencies loading time
    if cmd == "preprocess":
        from preprocessing import process_videos
        process_videos()

    elif cmd == "train":
        from train import CFG as TrainCFG, train as train_model
        if parsed.epochs: TrainCFG.max_epochs = parsed.epochs
        if parsed.batch_size: TrainCFG.batch_size = parsed.batch_size
        if parsed.lr: TrainCFG.learning_rate = parsed.lr
        train_model(temporal=False, resume=parsed.resume)

    elif cmd == "train_temporal":
        from train import CFG as TrainCFG, train as train_model
        if parsed.epochs: TrainCFG.max_epochs = parsed.epochs
        if parsed.batch_size: TrainCFG.batch_size = parsed.batch_size
        if parsed.lr: TrainCFG.learning_rate = parsed.lr
        train_model(temporal=True, resume=parsed.resume)

    elif cmd in ("webcam", "video", "image", "export", "ensemble"):
        _run_interface_command(cmd, argv, parsed.device, unknown)

    else:
        parser.print_help()


def _run_interface_command(cmd: str, argv: list, device: str, unknown: list = None):
    import interface as iface

    if cmd == "webcam":
        if len(argv) < 1:
            print("Usage: python main.py webcam <checkpoint.pt> [camera_id]")
            return
        ckpt = argv[0]
        cam = int(argv[1]) if len(argv) > 1 else 0
        model = iface.load_model(ckpt, temporal=False, device=device)
        iface.run_webcam(model, device, cam)

    elif cmd == "ensemble":
        if len(argv) < 2:
            print("Usage: python main.py ensemble <frame.pt> <temporal.pt> [--image img.jpg]")
            return
        f_ckpt, t_ckpt = argv[0], argv[1]
        f_model = iface.load_model(f_ckpt, temporal=False, device=device)
        t_model = iface.load_model(t_ckpt, temporal=True, device=device)
        print("Models loaded. Use --image flag to test.")
        
        img_path = None
        if unknown and "--image" in unknown:
            idx = unknown.index("--image")
            if idx + 1 < len(unknown):
                img_path = unknown[idx + 1]
                
        if img_path:
            frame = cv2.imread(img_path)
            det = iface.detect_faces_with_bbox(frame)
            for face, bbox in det:
                prob = iface.ensemble_predict(f_model, t_model, cv2.resize(face, (224, 224)), device)
                frame = iface.draw_overlay(frame, bbox, iface.CLASS_NAMES[prob.argmax()], prob.max())
                frame = iface.draw_panel(frame, prob)
            cv2.imshow("Ensemble", frame)
            cv2.waitKey(0)
            iface._close_windows_safe()


if __name__ == "__main__":
    # Windows platform support for compiled/frozen executables
    multiprocessing.freeze_support()
    main()
