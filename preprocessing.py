import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count


def preprocess_pure_face(face):
    """
    Standardizes face images with cropping, resizing, color conversion, and normalization.
    Keeps disk-saved .npy files pristine without fixed offline augmentations.
    """
    face = cv2.resize(face, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    face = (face - mean) / std
    face = np.transpose(face, (2, 0, 1))
    return face.astype(np.float32)


def process_single_video(args):
    import mediapipe as mp
    video_path, save_dir, frame_interval = args

    # Initialize MediaPipe per-process to avoid cross-process deadlocks
    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )

    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved_id = 0

    print(f"[Processing] {video_path}", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face.process(rgb)

            if results.detections:
                h, w = frame.shape[:2]
                for i, det in enumerate(results.detections):
                    bbox = det.location_data.relative_bounding_box
                    y_min = max(0, int(bbox.ymin * h))
                    x_min = max(0, int(bbox.xmin * w))
                    y_max = min(h, int((bbox.ymin + bbox.height) * h))
                    x_max = min(w, int((bbox.xmin + bbox.width) * w))
                    face = frame[y_min:y_max, x_min:x_max]

                    if face.size == 0:
                        continue

                    processed = preprocess_pure_face(face)
                    save_path = os.path.join(save_dir, f"{saved_id:05d}_{i}.npy")
                    np.save(save_path, processed)

            saved_id += 1

        frame_id += 1

    cap.release()
    mp_face.close()
    print(f"[Done] {video_path}", flush=True)


def process_videos(
    raw_root="dataset_raw",
    output_root="dataset_faces",
    frame_interval=5,
    num_workers=4
):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    tasks = []
    for cls in ["real", "deepfake", "filter"]:
        input_dir = os.path.join(raw_root, cls)
        if not os.path.exists(input_dir):
            print(f"[Skip] {input_dir} not found")
            continue

        for video_name in os.listdir(input_dir):
            if not video_name.endswith((".mp4", ".avi", ".mov")):
                continue
            video_path = os.path.join(input_dir, video_name)
            video_id = os.path.splitext(video_name)[0]
            save_dir = os.path.join(output_root, cls, video_id)
            tasks.append((video_path, save_dir, frame_interval))

    total = len(tasks)
    print(f"Found {total} videos, processing with {num_workers} parallel workers...")

    with Pool(processes=num_workers) as pool:
        pool.map(process_single_video, tasks)

    print(f"\nPreprocessing complete. Output directory: {output_root}")


if __name__ == "__main__":
    process_videos(
        raw_root="dataset_raw",
        output_root="dataset_faces",
        frame_interval=5,
        num_workers=4
    )
