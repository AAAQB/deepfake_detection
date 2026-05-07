"""
download_dataset.py — Dataset preparation for deepfake detection.

PDF Section 5: Dataset Collection
- FaceForensics++, Celeb-DF, DFDC — referenced sources
- Synthetic Filter Dataset — self-generated

This script does NOT generate synthetic data. It provides:
  1. Status check: scan dataset_raw/ and dataset_faces/ for existing data
  2. Placeholder instructions for downloading standard datasets
  3. A dry-run preprocessing validator
"""

import os
import glob


RAW_DIR = "dataset_raw"
FACE_DIR = "dataset_faces"
CLASS_NAMES = ["real", "filter", "deepfake"]


def check_dataset_status():
    """Scan existing data folders and report what's available."""
    print("=" * 55)
    print("  Dataset Status Report")
    print("=" * 55)

    total_raw = 0
    total_faces = 0

    for cls in CLASS_NAMES:
        raw_cls = os.path.join(RAW_DIR, cls)
        face_cls = os.path.join(FACE_DIR, cls)

        raw_files = []
        if os.path.isdir(raw_cls):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.mp4", "*.avi", "*.mov", "*.mkv"):
                raw_files.extend(glob.glob(os.path.join(raw_cls, "**", ext), recursive=True))

        face_count = 0
        if os.path.isdir(face_cls):
            for root, _, files in os.walk(face_cls):
                face_count += sum(1 for f in files if f.endswith(".npy"))

        status_raw = f"{len(raw_files)} files" if raw_files else "empty"
        status_face = f"{face_count} crops" if face_count else "empty"
        print(f"  {cls:>8}  │ raw: {status_raw:>12}  │ faces: {status_face:>12}")
        total_raw += len(raw_files)
        total_faces += face_count

    print("-" * 55)
    print(f"  Total:      {total_raw:>5} raw files, {total_faces:>5} face crops")

    if total_raw == 0 and total_faces == 0:
        print()
        print("  ⚠️  No data found. Place your data in:")
        print("     dataset_raw/real/       — real faces")
        print("     dataset_raw/filter/     — filtered/AR faces")
        print("     dataset_raw/deepfake/   — deepfake faces")
        print()
        print("  Supported formats: .jpg, .jpeg, .png, .mp4, .avi, .mov, .mkv")
        print("  Then run:  python preprocessing.py")
        print("  This will extract and normalize face crops into dataset_faces/")


def download_faceforensics(target_dir="dataset_raw"):
    """Instructions for FaceForensics++ dataset."""
    print("FaceForensics++ download instructions:")
    print(f"  1. Visit: https://github.com/ondyari/FaceForensics")
    print("  2. Request access (academic license)")
    print(f"  3. Download and extract frames into: {os.path.abspath(target_dir)}/real/")
    print(f"     and manipulated into:             {os.path.abspath(target_dir)}/deepfake/")


def download_celebdf(target_dir="dataset_raw"):
    """Instructions for Celeb-DF dataset."""
    print("Celeb-DF download instructions:")
    print("  1. Visit: https://github.com/yuezunli/Celeb-DF")
    print("  2. Follow access instructions")
    print(f"  3. Extract into: {os.path.abspath(target_dir)}/deepfake/ (v2)")


def download_dfdc(target_dir="dataset_raw"):
    """Instructions for DFDC dataset."""
    print("DFDC (DeepFake Detection Challenge) download instructions:")
    print("  1. Visit: https://www.kaggle.com/c/deepfake-detection-challenge")
    print("  2. Download the dataset from Kaggle")
    print(f"  3. Extract into: {os.path.abspath(target_dir)}/")


def validate_preprocessing_ready():
    """Check that the data is ready for preprocessing.py to run."""
    has_any = False
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(RAW_DIR, cls)
        if os.path.isdir(cls_dir):
            files = [f for f in os.listdir(cls_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv"))]
            if files:
                print(f"  ✅ {cls}: {len(files)} files ready")
                has_any = True
            else:
                print(f"  ⚠️  {cls}: directory exists but no supported files found")
        else:
            print(f"  ❌ {cls}: directory missing — create dataset_raw/{cls}/ and add data")

    if has_any:
        print()
        print("  Run:  python preprocessing.py  to extract faces")
    else:
        print()
        print("  No data found. Add real/filter/deepfake images or videos first.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset tools (PDF §5)")
    parser.add_argument("--mode", type=str,
                        choices=["status", "faceforensics", "celebdf", "dfdc", "validate"],
                        default="status",
                        help="status=check folders | validate=pre-check preprocessing | * = download instructions")
    parser.add_argument("--output_dir", type=str, default="dataset_raw")
    args = parser.parse_args()

    if args.mode == "status":
        check_dataset_status()
    elif args.mode == "validate":
        validate_preprocessing_ready()
    elif args.mode == "faceforensics":
        download_faceforensics(args.output_dir)
    elif args.mode == "celebdf":
        download_celebdf(args.output_dir)
    elif args.mode == "dfdc":
        download_dfdc(args.output_dir)
