
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import EfficientNet as EN
import LSTM as LS
from facedetector import create_detector, detect_faces, preprocess_face

CLASS_NAMES = ["REAL", "FILTER", "DEEPFAKE"]
CLASS_COLORS = {
    0: (0, 220, 80),
    1: (0, 180, 255),
    2: (40, 40, 255),
}


def main():
    parser = argparse.ArgumentParser(description="Real-Time Deepfake Detection")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best.pt checkpoint")
    parser.add_argument("--model_type", type=str, default="efficientnet",
                        choices=["efficientnet", "temporal"],
                        help="Model architecture (PDF Section 8 or 10)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID")
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available() and not args.no_amp

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    if args.model_type == "efficientnet":
        model = EN.EfficientNet(num_classes=3)
    else:
        model = LS.CNNLSTM(num_classes=3, seq_len=8)

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print(f"Loaded {args.model_type.upper()} from {args.checkpoint}")

    # MediaPipe face detector (via shared module)
    face_detector = create_detector(min_confidence=0.5)

    # Webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # FPS
    fps = 0
    frame_count = 0
    fps_timer = time.perf_counter()

    # LSTM buffer
    is_temporal = args.model_type == "temporal"
    frame_buffer = []

    print("\n" + "=" * 50)
    print("Real-Time Detection Running (PDF Section 11)")
    print("  [q] Quit")
    print("=" * 50)

    cv2.namedWindow("DeepGuard Realtime", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Detect faces
            faces = detect_faces(face_detector, frame)

            # Process each face
            for face_info in faces:
                x1, y1, x2, y2 = face_info["bbox"]
                x1m, y1m, x2m, y2m = face_info["expanded_bbox"]
                face = frame[y1m:y2m, x1m:x2m]
                if face.size == 0:
                    continue

                # Inference
                tensor = torch.from_numpy(preprocess_face(face))

                if is_temporal:
                    frame_buffer.append(tensor)
                    if len(frame_buffer) < 8:
                        continue
                    # Keep sliding window: only use the latest 8 frames
                    window = frame_buffer[-8:]
                    seq = torch.cat(window, dim=0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=amp):
                            output = model(seq)
                else:
                    tensor = tensor.to(device)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=amp):
                            output = model(tensor)

                probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
                pred_class = int(np.argmax(probs))
                confidence = probs[pred_class]
                color = CLASS_COLORS[pred_class]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Draw label above box
                label = f"{CLASS_NAMES[pred_class]} ({confidence:.2f})"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y1 - 10, 25)
                cv2.rectangle(frame, (x1, label_y - lh - 6),
                              (x1 + lw + 8, label_y + 4), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1 + 4, label_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Confidence bar
                bar_width = int(confidence * (x2 - x1))
                bar_y = y2 + 8
                cv2.rectangle(frame, (x1, bar_y), (x1 + bar_width, bar_y + 6), color, -1)
                cv2.putText(frame, f"{confidence:.0%}",
                            (x1 + bar_width + 4, bar_y + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # FPS overlay
            frame_count += 1
            if time.perf_counter() - fps_timer >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_timer = time.perf_counter()

            cv2.putText(frame, f"FPS: {fps} | {args.model_type.upper()}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.imshow("DeepGuard Realtime", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Inference stopped.")


if __name__ == "__main__":
    main()
