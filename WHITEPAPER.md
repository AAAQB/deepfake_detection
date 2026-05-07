# DeepGuard: A Real-Time Facial Manipulation Detection System

## Technical Specification

**Version:** 2.0  
**Date:** 2026-05-07  
**Classification:** Technical Reference Document

---

## 1. System Overview

DeepGuard is a real-time facial manipulation detection system that classifies video faces into three categories: unmodified (real), beauty-filter-modified (filter), and deepfake-generated (deepfake). The system implements a dual-architecture approach supporting both single-frame classification (EfficientNet-B4) and temporal sequence analysis (CNN-BiLSTM).

### 1.1 Processing Pipeline

The inference pipeline follows a sequential architecture:

```
Video/Camera Input
    → Frame Capture (OpenCV)
    → Face Detection (MediaPipe BlazeFace, ss 0.5)
    → Face Crop & Resize (224×224)
    → Normalization (ImageNet statistics)
    → Neural Network Inference
    → Softmax (3-class)
    → Overlay Rendering
```

### 1.2 Execution Modes

The system supports three execution modes:
- **Real-time camera inference** (`inference.py`)
- **Video/image file inference** (`inference.py` with `--source` support)
- **Batch training** (`train.py`)

---

## 2. Model Architecture

### 2.1 Single-Frame Classifier (EfficientNet-B4)

**Source:** `EfficientNet.py`

**Architecture:**
- Backbone: EfficientNet-B4, pretrained on ImageNet-1K
- Input dimensions: (B, 3, 224, 224)
- Output dimensions: (B, 3) — unnormalized logits

**Classification head:**
```
Dropout(p=dropout)
Linear(1792, 512)
ReLU()
Dropout(p=dropout / 2)
Linear(512, 3)
```

**Parameter count:** ~19M total

**Configuration parameters:**
- `num_classes` (default: 3) — output class count
- `dropout` (default: 0.4) — dropout rate applied before each linear layer
- `freeze_backbone` (default: False) — when enabled, gradients are not computed for EfficientNet-B4 parameters; only the classification head is trained
- `pretrained` (default: True) — initialize backbone with ImageNet-1K weights

**Feature extraction:** `get_embedding(x)` returns the 1792-dimensional feature vector from the average pooling layer, prior to classification. This supports downstream feature analysis or ensemble integration.

### 2.2 Temporal Sequence Classifier (CNN-BiLSTM)

**Source:** `LSTM.py`

**Architecture:**
- Visual backbone: EfficientNet-B4 (shared across frames)
- Sequence processor: 2-layer Bidirectional LSTM
- Input dimensions: (B, T, 3, 224, 224) where T = sequence length
- Output dimensions: (B, 3) — unnormalized logits

**Feature extraction pipeline:**
```
Input (B, T, 3, 224, 224)
    → Reshape to (B×T, 3, 224, 224)
    → EfficientNet-B4 features (B×T, 1792)
    → Reshape to (B, T, 1792)
    → BiLSTM (B, T, lstm_out_dim)
    → Mean pooling over T (B, lstm_out_dim)
    → Classification head (B, 3)
```

**LSTM configuration:**
- `hidden_size` (default: 512)
- `num_layers` (default: 2)
- `bidirectional` (default: True)
- LSTM output dimension: `hidden_size × 2 = 1024`
- Inter-layer dropout: 0.4 (applied when `num_layers > 1`)

**Mean pooling rationale:** The temporal output `lstm_out` is aggregated via mean over the sequence dimension rather than selecting the final output. This provides equal weighting across all frames in the window, which is appropriate for detecting temporally distributed artifacts.

**Sequence embedding extraction:** `extract_sequence_embeddings(x)` returns the per-frame feature vectors of shape (B, T, 1792) prior to LSTM processing.

### 2.3 Data Flow

```
Single-frame path:
  224×224 crop → Normalize → EfficientNet-B4 → Logits (3)

Temporal path:
  [frame₁, frame₂, ..., frame₈] → Normalize → EfficientNet-B4 (per frame) →
  Stack (B, 8, 1792) → BiLSTM → Mean Pool → Logits (3)
```

---

## 3. Data Processing Pipeline

### 3.1 Face Detection (MediaPipe BlazeFace)

**Source:** `facedetector.py`

**Detection model:** BlazeFace Short Range (float16 TFLite, 229 KB)

**Parameters:**
- `min_detection_confidence`: 0.5
- Running mode: IMAGE (single-frame)
- Face margin expansion: 15% of bounding box width/height on each side

**Output structure per detection:**
```python
{
    "bbox": (x1, y1, x2, y2),           # original detection
    "expanded_bbox": (x1m, y1m, x2m, y2m),  # expanded by margin
    "score": float                       # detection confidence
}
```

**Minimum face area filter:** 1000 pixels (to reject false positives).

### 3.2 Face Normalization

**Normalization pipeline (training and inference):**
```
BGR → RGB → float32 / 255.0 → (x - μ) / σ → CHW transpose
```
Where `μ = [0.485, 0.456, 0.406]` and `σ = [0.229, 0.224, 0.225]` (ImageNet statistics).

**Size:** 224 × 224 pixels (bilinear interpolation).

Two paths handle normalization:
1. **Preprocessing for training** (`_normalize_and_save`): saves normalized tensor as `.npy` to disk
2. **Inference** (`preprocess_face`): returns normalized tensor with batch dimension (1, 3, 224, 224)

### 3.3 Dataset Structure

**Directory format:**
```
dataset_raw/
  real/       — unmodified face images/videos
  filter/     — beauty/AR filter modified faces
  deepfake/   — deepfake generated faces

dataset_faces/  — after preprocessing.py extraction
  real/
    {video_id}/
      00000.npy
      00001.npy
      ...
  filter/
    {video_id}/
      ...
  deepfake/
    {video_id}/
      ...
```

**Label encoding:** `0 → real`, `1 → filter`, `2 → deepfake` (enforced by directory structure).

### 3.4 Face Extraction Pipeline

**Source:** `preprocessing.py`

**Supported input formats:** .jpg, .jpeg, .png, .mp4, .avi, .mov, .mkv

**Processing parameters:**
- `frame_interval`: 5 (extract every 5th frame from videos)
- `target_size`: 224

**Edge cases:**
- Multiple faces per frame: each detected face is extracted as a separate crop
- No face detected in image: the full image is resized to 224×224 and saved as a fallback
- No faces detected in video frame: frame is silently skipped

---

## 4. Data Augmentation

**Source:** `dataset.py`, method `_augment`

Applied at runtime during training with the following probabilities and operations:

| Operation | Probability | Parameters | Implementation |
|-----------|-----------|------------|----------------|
| Horizontal flip | 0.5 | — | `torch.flip(dims=[-1])` |
| Rotation | 0.3 | ±10°, bilinear | Affine grid + `grid_sample`, fill=0 |
| Random zoom | 0.3 | crop 85-100% of area, resize back | `F.interpolate`, bilinear, `align_corners=False` |
| Brightness | 0.4 | factor 0.7-1.3 | Element-wise multiplication |
| Gaussian blur | 0.25 | kernel 3 or 5, σ ≈ 0.8-1.4 | Separable 1D kernel convolution, group=c |
| JPEG compression | 0.2 | quality 50-90 | PIL JPEG encode → decode via BytesIO, subsampling=0 |
| Color jitter | 0.3 | contrast 0.8-1.2, saturation 0.8-1.2 | Per-channel contrast → luminance saturation |

**Output clamping:** Augmented tensor is clamped to [-3.0, 3.0] (approximately ±3σ of normalized pixel values).

**JPEG compression note:** The compression simulation performs an actual JPEG codec round-trip via PIL (`subsampling=0` to preserve chroma channels), producing authentic DCT-based block artifacts rather than synthetic quantization noise.

---

## 5. Training Procedure

**Source:** `train.py`

### 5.1 Configuration

| Parameter | Default | Range |
|-----------|---------|-------|
| `--model_type` | efficientnet | efficientnet, temporal |
| `--batch_size` | 32 | dependent on GPU memory |
| `--epochs` | 30 | 1–∞ |
| `--lr` | 1e-3 | — |
| `--lr_min` | 1e-5 | — |
| `--weight_decay` | 1e-4 | — |
| `--dropout` | 0.4 | 0.0–1.0 |
| `--hidden_size` | 512 | temporal model only |
| `--seq_len` | 8 | temporal model only |
| `--val_split` | 0.15 | — |
| `--patience` | 10 | early stopping |
| `--num_workers` | 4 | data loading threads |
| `--freeze_backbone` | False | freeze CNN feature extractor |

### 5.2 Optimizer and Loss

- **Optimizer:** AdamW (`betas=(0.9, 0.999)`, default eps)
- **Loss function:** Cross-entropy (`nn.CrossEntropyLoss`, no class weighting)
- **Gradient clipping:** max norm 5.0

### 5.3 Learning Rate Schedule

Cosine annealing from `lr` to `lr_min` over `epochs` iterations:
```
η(t) = η_min + (η_0 - η_min) × 0.5 × (1 + cos(π × t / T_max))
```

### 5.4 Mixed Precision Training

Conditionally enabled when CUDA is available and `--no_amp` is not set:
- `torch.cuda.amp.autocast` for forward pass
- `GradScaler` for gradient scaling
- Fallback to FP32 when AMP is disabled or running on CPU

### 5.5 Optimizer Configuration with Frozen Backbone

When `--freeze_backbone` is set, only the classification head parameters are passed to the optimizer:
```python
if args.freeze_backbone:
    optimizer = optim.AdamW(model.classifier.parameters(), ...)
```

### 5.6 Checkpointing

Checkpoints are saved to `checkpoints/{model_type}_{timestamp}/`:
- `last.pt` — most recent epoch
- `best.pt` — epoch with highest validation accuracy
- `args.json` — training configuration

Each checkpoint contains: `epoch`, `model_state_dict`, `optimizer_state_dict`, `val_acc`, `train_acc`, `args`.

### 5.7 Early Stopping

Training terminates when validation accuracy has not improved for `patience` consecutive epochs.

---

## 6. Inference Pipeline

**Source:** `inference.py`

### 6.1 Execution Flow

1. Initialize camera (OpenCV, 640×480, horizontal mirror)
2. Load model checkpoint
3. Initialize MediaPipe face detector (min_confidence=0.5)
4. Loop:
   a. Read frame
   b. Detect faces
   c. For each face: crop → normalize → model inference → softmax
   d. Render bounding box, label, confidence bar
   e. Calculate and display FPS
   f. Exit on 'q' key

### 6.2 Model Loading

Architecture is determined from checkpoint metadata (`args.model_type`):
- `efficientnet` → `EfficientNet()` with default parameters
- `temporal` → `CNNLSTM()` with `hidden_size=512` and `seq_len=8`

### 6.3 Temporal Inference Buffer

The temporal model maintains a sliding window of the 8 most recent face crops. Each frame appends to the buffer; inference is triggered only when the buffer contains at least 8 frames. The window is truncated to the last 8 frames before each inference step.

### 6.4 FPS Calculation

FPS is computed as frame count over a 1-second sliding window, reported on the overlay.

---

## 7. Model Export (ONNX)

**Source:** `train.py`, post-training export

```python
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
)
```

- **Input shape:** (1, 3, 224, 224) for EfficientNet; (1, 8, 3, 224, 224) for temporal
- **Dynamic axis:** batch size (axis 0) is dynamic
- **Output path:** same directory as checkpoint

---

## 8. Evaluation

**Source:** `evaluate.py`

### 8.1 Metrics

| Metric | Implementation | Scope |
|--------|---------------|-------|
| Accuracy | `sklearn.metrics.accuracy_score` | Global |
| Precision (weighted) | `sklearn.metrics.precision_recall_fscore_support(average="weighted")` | Global |
| Recall (weighted) | Same function | Global |
| F1 (weighted) | Same function | Global |
| Precision (per-class) | `average=None` | Per class |
| Recall (per-class) | Same function | Per class |
| F1 (per-class) | Same function | Per class |
| AUC (per-class) | `roc_auc_score` (One-vs-Rest) | Per class |
| Confusion matrix | `confusion_matrix` | All classes |

### 8.2 Outputs

Results are saved to `evaluation_results/{checkpoint_name}/`:
- `{name}_metrics.json` — full metric dump
- `{name}_confusion.png` — normalized confusion matrix heatmap
- `{name}_roc.png` — ROC curves (One-vs-Rest)

### 8.3 AMP Behavior

Mixed precision is enabled during evaluation when CUDA is available. Disabled with `--no_amp`.

---

## 9. Dependencies

**Source:** `requirements.txt`

| Package | Version Constraint | Purpose |
|---------|-------------------|---------|
| torch | ≥2.0.0 | Model definition, training, inference |
| torchvision | ≥0.15.0 | Pretrained model weights |
| opencv-python | ≥4.8.0 | Video I/O, image processing |
| mediapipe | ≥0.10.0 | Face detection (tasks API) |
| numpy | ≥1.24.0 | Array operations |
| scikit-learn | ≥1.3.0 | Evaluation metrics |
| matplotlib | ≥3.7.0 | ROC/confusion matrix plotting |
| seaborn | ≥0.12.0 | Heatmap visualization |
| tensorboard | ≥2.13.0 | Training logging |
| tqdm | ≥4.65.0 | Progress bars |
| pillow | ≥10.0.0 | JPEG compression simulation |

---

## 10. Hardware Requirements

**Minimum (inference, CPU):**
- CPU with AVX2 support
- 8 GB system memory
- Camera (USB or integrated)

**Recommended (training, GPU):**
- GPU with ≥8 GB VRAM (e.g., NVIDIA RTX 4070)
- 16 GB system memory
- CUDA 11.8+ / cuDNN 8+

**Cloud environments:** Functionally equivalent to the recommended configuration on Google Colab or Kaggle with GPU runtime enabled.

---

## 11. Command Reference

### Dataset status
```
python download_dataset.py --mode status
python download_dataset.py --mode validate
```

### Face extraction
```
python preprocessing.py --raw_root dataset_raw --output_root dataset_faces
```

### Training
```
python train.py --model_type efficientnet --epochs 30 --batch_size 32
python train.py --model_type temporal --epochs 40 --hidden_size 512 --seq_len 8 --freeze_backbone
```

### Inference
```
python inference.py --checkpoint checkpoints/efficientnet_*/best.pt --model_type efficientnet
```

### Evaluation
```
python evaluate.py --checkpoint checkpoints/efficientnet_*/best.pt
```

### ONNX export
ONNX export occurs automatically at the end of training. To export-only:
```
python -c "import torch; m = torch.load('checkpoints/.../best.pt', map_location='cpu')['model_state_dict']; ..."
```

---

## 12. Known Limitations

1. **Single face priority:** The inference pipeline processes each detected face independently; multi-face scenarios produce concurrent classification outputs but no face tracking or re-identification between frames.

2. **Temporal model buffer persistence:** The sliding window approach provides per-instance temporal context but does not maintain identity tracking across multiple faces in the same frame.

3. **Filter detection bound:** Beauty filter deception effectiveness is proportional to filter intensity. Subtle filters (low opacity, minimal morphological changes) may produce feature distributions overlapping with real faces.

4. **Data distribution dependency:** Classification accuracy is contingent on the distribution of the training dataset. Out-of-distribution samples (novel generation methods, unseen camera pipelines, different resolutions) may degrade performance below training-set metrics.
