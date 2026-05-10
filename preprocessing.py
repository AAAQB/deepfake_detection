import os
import cv2
import mediapipe as mp
import numpy as np

# =========================
# Initialization
# =========================
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)
 
# =========================
# Video Frame Extraction
# =========================
def extract_frames_from_video(video_path, output_dir, frame_interval=5, downscale=0.5):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lower resolution first
        h, w = frame.shape[:2]

        frame = cv2.resize(
            frame,
            (int(w * downscale), int(h * downscale))
        )
        # =======================

        if frame_id % frame_interval == 0:
            filename = os.path.join(
                output_dir,
                f"frame_{saved_id:05d}.jpg"
            )

            cv2.imwrite(filename, frame)
            saved_id += 1

        frame_id += 1


    cap.release()


# =========================
# Face detection
# =========================
def extract_faces_from_image(image):
    # Use pixel indices directly
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    faces = []

    if results.detections:
        h, w, _ = image.shape

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

        
            y_min, x_min = int(bbox.ymin * h), int(bbox.xmin * w)
            y_max, x_max = int((bbox.ymin + bbox.height) * h), int((bbox.xmin + bbox.width) * w)

            
            y_min, x_min = max(0, y_min), max(0, x_min)
            y_max, x_max = min(h, y_max), min(w, x_max)

            # extract by pixel
            face = image[y_min:y_max, x_min:x_max]

            if face.size > 0:
                faces.append(face)

    return faces


# =========================
# Data Augmentation
# =========================

def random_horizontal_flip(img, p=0.5):
    if np.random.rand() < p:
        img = cv2.flip(img, 1)
    return img


def random_brightness(img, factor=0.2):
    alpha = 1.0 + np.random.uniform(-factor, factor)
    img = np.clip(img * alpha, 0, 255).astype(np.uint8)
    return img


def random_compression(img, quality_range=(30, 100), p=0.3):
    if np.random.rand() < p:
        quality = np.random.randint(*quality_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)
    return img
#

# =========================
# Preprocessing（train）
# =========================
def preprocess_train(face):
    """
    input: BGR face
    output: tensor(numpy) after normalization
    """

    # Resize
    face = cv2.resize(face, (224, 224))

    # Data Augmentation
    face = random_horizontal_flip(face)
    face = random_brightness(face)
    face = random_compression(face)

    # to RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Normalize 
    face = face / 255.0
    mean = np.array([0.485, 0.456, 0.406],dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225],dtype=np.float32)
    face = (face - mean) / std

    # HWC to CHW
    face = np.transpose(face, (2, 0, 1))

    return face.astype(np.float32)


# =========================
# Preprocessing（infer）
# =========================
def preprocess_infer(face):
    face = cv2.resize(face, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face = face / 255.0
    mean = np.array([0.485, 0.456, 0.406],dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225],dtype=np.float32)
    face = (face - mean) / std

    face = np.transpose(face, (2, 0, 1))
    return face.astype(np.float32)


# =========================
# Processing
# =========================
def process_videos(
    raw_root="dataset_raw",
    output_root="dataset_faces",
    frame_interval=5,
    mode="train"  # train / infer
):
    for cls in ["real", "deepfake", "filter"]:
        input_dir = os.path.join(raw_root, cls)

        for video_name in os.listdir(input_dir):
            if not video_name.endswith((".mp4", ".avi", ".mov")):
                continue

            video_path = os.path.join(input_dir, video_name)
            video_id = os.path.splitext(video_name)[0]

            save_dir = os.path.join(output_root, cls, video_id)
            os.makedirs(save_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            frame_id = 0
            saved_id = 0

            print(f"[Processing] {video_path}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % frame_interval == 0:
                    faces = extract_faces_from_image(frame)

                    for i, face in enumerate(faces):
                        if mode == "train":
                            processed = preprocess_train(face)
                        else:
                            processed = preprocess_infer(face)

                        save_path = os.path.join(
                            save_dir,
                            f"{saved_id:05d}_{i}.npy"
                        )
                        np.save(save_path, processed)

                    saved_id += 1

                frame_id += 1

            cap.release()


# =========================
# Entrance
# =========================
if __name__ == "__main__":
    process_videos(
        raw_root="dataset_raw",
        output_root="dataset_faces",
        frame_interval=5,
        mode="train"
    )