"""
All detection pipeline files import from here for consistent behavior:
- preprocessing.py (face extraction for training)
- inference.py (real-time camera)

Usage:
    from facedetector import create_detector, detect_faces
    detector = create_detector(min_confidence=0.5)
    faces = detect_faces(detector, frame_bgr)  # returns list of relative bboxes
"""
import os
import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode

MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def ensure_model():
    """Download the MediaPipe face detection model if not present."""
    if not os.path.exists(MODEL_PATH):
        from urllib.request import urlretrieve
        print("Downloading face detection model...")
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/"
            "face_detector/blaze_face_short_range/float16/latest/"
            "blaze_face_short_range.tflite",
            MODEL_PATH,
        )
        print("Downloaded:", MODEL_PATH)


def create_detector(min_confidence=0.5):
    """Create a MediaPipe FaceDetector instance."""
    ensure_model()
    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = FaceDetectorOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        min_detection_confidence=min_confidence,
    )
    return FaceDetector.create_from_options(options)


def _normalize_and_save(face_bgr, save_path, target_size=224):
    """Normalize a face crop and save as .npy (for preprocessing pipeline)."""
    face = cv2.resize(face_bgr, (target_size, target_size))
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    face_rgb = (face_rgb - IMAGENET_MEAN) / IMAGENET_STD
    face_rgb = np.transpose(face_rgb, (2, 0, 1)).astype(np.float32)
    np.save(save_path, face_rgb)


def preprocess_face(face_bgr, target_size=224):
    """Normalize a face crop to a PyTorch input tensor (for inference)."""
    face = cv2.resize(face_bgr, (target_size, target_size))
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    face_rgb = (face_rgb - IMAGENET_MEAN) / IMAGENET_STD
    face_rgb = np.transpose(face_rgb, (2, 0, 1))
    return np.expand_dims(face_rgb, axis=0)


def detect_faces(detector, frame_bgr, min_area=1000, expand_margin=0.15):
    """
    Run MediaPipe face detection on a BGR frame.

    Returns list of dicts:
        { "bbox": (x1, y1, x2, y2), "expanded_bbox": (x1m, y1m, x2m, y2m) }
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    faces = []
    if result.detections:
        for det in result.detections:
            bbox = det.bounding_box
            x1 = max(0, bbox.origin_x)
            y1 = max(0, bbox.origin_y)
            x2 = min(w, bbox.origin_x + bbox.width)
            y2 = min(h, bbox.origin_y + bbox.height)
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue

            margin = int((x2 - x1) * expand_margin)
            x1m = max(0, x1 - margin)
            y1m = max(0, y1 - margin)
            x2m = min(w, x2 + margin)
            y2m = min(h, y2 + margin)

            faces.append({
                "bbox": (x1, y1, x2, y2),
                "expanded_bbox": (x1m, y1m, x2m, y2m),
                "score": det.categories[0].score if det.categories else 1.0,
            })

    return faces
