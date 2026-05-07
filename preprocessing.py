

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

from facedetector import create_detector, detect_faces, _normalize_and_save

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

SUPPORTED_EXT = (".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png")

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = create_detector(min_confidence=0.5)
    return _detector


def extract_faces(video_path, output_dir, frame_interval=5, target_size=224):
    """Extract face crops from a video file."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    detector = get_detector()

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        faces = detect_faces(detector, frame)
        for face in faces:
            x1m, y1m, x2m, y2m = face["expanded_bbox"]
            crop = frame[y1m:y2m, x1m:x2m]
            if crop.size == 0:
                continue
            save_path = os.path.join(save_dir, f"{saved_count:05d}.npy")
            _normalize_and_save(crop, save_path, target_size)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"  {video_name}: {saved_count} faces")
    return saved_count


def extract_single_image(image_path, output_dir, video_name, target_size=224):
    """Extract faces from a single image."""
    save_dir = os.path.join(output_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    detector = get_detector()
    frame = cv2.imread(image_path)
    if frame is None:
        return 0

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    saved_count = 0

    faces = detect_faces(detector, frame)
    for face in faces:
        x1m, y1m, x2m, y2m = face["expanded_bbox"]
        crop = frame[y1m:y2m, x1m:x2m]
        if crop.size == 0:
            continue
        save_path = os.path.join(save_dir, f"{base_name}_{saved_count:03d}.npy")
        _normalize_and_save(crop, save_path, target_size)
        saved_count += 1

    # Fallback: use full image if no face detected
    if saved_count == 0:
        save_path = os.path.join(save_dir, f"{base_name}_000.npy")
        _normalize_and_save(cv2.resize(frame, (target_size, target_size)), save_path, target_size)
        saved_count = 1

    return saved_count


def process_all_videos(raw_root="dataset_raw", output_root="dataset_faces",
                       frame_interval=5):
    """Process all media in raw_root/{real,filter,deepfake}/..."""
    for cls_name in ["real", "deepfake", "filter"]:
        input_dir = os.path.join(raw_root, cls_name)
        if not os.path.isdir(input_dir):
            print(f"  Warning: {input_dir} not found, skipping")
            continue

        media_files = []
        for ext in SUPPORTED_EXT:
            media_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
            media_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))

        image_files = [f for f in media_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        video_files = [f for f in media_files if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

        if not media_files:
            print(f"  No supported files in {input_dir}")
            continue

        cls_output = os.path.join(output_root, cls_name)
        total = 0

        if image_files:
            vid_name = "__images__"
            for img_path in tqdm(image_files, desc=f"Images ({cls_name})"):
                total += extract_single_image(img_path, cls_output, vid_name, target_size=224)

        if video_files:
            for vp in tqdm(video_files, desc=f"Videos ({cls_name})"):
                total += extract_faces(vp, cls_output, frame_interval)

        print(f"  {cls_name}: {total} faces total")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Face extraction (PDF §6)")
    parser.add_argument("--raw_root", default="dataset_raw")
    parser.add_argument("--output_root", default="dataset_faces")
    parser.add_argument("--frame_interval", type=int, default=5)
    args = parser.parse_args()

    print("Starting face extraction with MediaPipe...")
    process_all_videos(args.raw_root, args.output_root, args.frame_interval)
    print("Done.")
