# Deepfake Detection

An open-source AI system for detecting facial manipulation in images, videos, and real-time webcam streams. The system classifies faces into three categories:

- **Real** — Authentic unmodified faces
- **Filter** — Faces modified with beauty or AR filters
- **Deepfake** — AI-generated or manipulated faces

## Architecture

The system employs a dual-model ensemble architecture:

| Model | Backbone | Purpose |
|---|---|---|
| **Frame Model** | EfficientNet-B4 (ImageNet pretrained) | Classifies individual face images |
| **Temporal Model** | EfficientNet-B4 + Bidirectional LSTM | Analyzes sequential video frames for temporal inconsistencies |

### Key Features

- **Hybrid Detection** — Combines spatial (per-frame) and temporal (sequence-based) analysis
- **Real-Time Inference** — Webcam mode with live face detection and overlay visualization
- **Ensemble Prediction** — Fuses frame-level and temporal model outputs for improved accuracy
- **Grad-CAM Visualization** — Model interpretability via activation heatmaps
- **Robustness Evaluation** — Tests model performance under noise, blur, compression, and other distortions
- **ONNX Export** — Model deployment support via ONNX format

## Project Structure

```
├── main.py                    # CLI entry point
├── deepfake_pipeline.py       # End-to-end pipeline (single-file version)
├── train.py                   # Training logic with full config
├── preprocessing.py           # Face extraction & normalization
├── EfficientNet.py            # EfficientNet-B4 classifier module
├── LSTM.py                    # Temporal (EfficientNet + LSTM) module
├── interface.py               # Inference, visualization, and webcam interface
├── evaluation.py              # Evaluation metrics, confusion matrix, ROC curves
├── robustness.py              # Robustness testing under image corruptions
├── grad_cam.py                # Grad-CAM heatmap generation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Requirements

- Python 3.9+
- CUDA-capable GPU recommended (training & real-time inference)
- Webcam for live detection mode

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Organize your dataset in the following structure:

```
data_raw/
├── image/
│   ├── real/
│   ├── filter/
│   └── deepfake/
└── video/
    ├── real/
    ├── filter/
    └── deepfake/
```

Supported formats:
- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- **Videos**: `.mp4`, `.avi`, `.mov`, `.mkv`

## Usage

### 1. Preprocessing

Extract and normalize faces from raw images and videos:

```bash
python main.py preprocess
```

Output is saved to `data_face/` as normalized `.npy` files (224×224, ImageNet-standardized, CHW format).

### 2. Training

**Frame model** (EfficientNet-B4):

```bash
python main.py train
```

**Temporal model** (EfficientNet-B4 + LSTM):

```bash
python main.py train_temporal
```

**Override hyperparameters**:

```bash
python main.py train --epochs 50 --batch-size 32 --lr 1e-4
python main.py train_temporal --epochs 30 --batch-size 8
```

**Resume from checkpoint**:

```bash
python main.py train --resume checkpoints/frame_epoch_20.pt
```

Checkpoints are saved to `checkpoints/` and training logs to `logs/` (viewable via TensorBoard).

### 3. Inference

**Webcam (real-time)**:

```bash
python main.py webcam checkpoints/frame_best_model.pt [camera_id]
```

**Single image**:

```bash
python main.py image checkpoints/frame_best_model.pt path/to/image.jpg
```

**Video file**:

```bash
python main.py video checkpoints/frame_best_model.pt path/to/video.mp4
```

**Ensemble (frame + temporal)**:

```bash
python main.py ensemble checkpoints/frame_best_model.pt checkpoints/temporal_best_model.pt --image path/to/image.jpg
```

### 4. Evaluation

```bash
python evaluation.py
```

Generates classification reports, confusion matrices, and ROC curves.

### 5. Robustness Testing

```bash
python robustness.py
```

Evaluates model performance under Gaussian noise, blur, brightness/contrast changes, rotation, and JPEG compression.

### 6. Model Interpretability

```bash
python grad_cam.py
```

Produces Grad-CAM heatmaps overlaying model attention regions on input faces.

### 7. ONNX Export

```bash
python main.py export checkpoints/frame_best_model.pt
```

## Configuration

All training hyperparameters are defined in the `CFG` dataclass within `train.py` and `deepfake_pipeline.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `backbone` | `efficientnet_b4` | CNN backbone architecture |
| `max_epochs` | `40` | Maximum training epochs |
| `batch_size` | `64` | Frame model batch size |
| `temporal_batch_size` | `8` | Temporal model batch size |
| `learning_rate` | `1e-4` | Initial learning rate |
| `hidden_size` | `512` | LSTM hidden dimension |
| `num_lstm_layers` | `4` | Number of LSTM layers |
| `temporal_seq_len` | `16` | Frames per video clip |
| `label_smoothing` | `0.1` | Label smoothing factor |
| `amp` | `auto` | Automatic mixed precision (enabled if CUDA available) |

## Visualization

During webcam/video inference, the interface displays:

- **Bounding boxes** color-coded by class (green = real, orange = filter, red = deepfake)
- **Confidence bar** below each detection
- **Prediction panel** showing per-class probabilities
- **FPS counter** for performance monitoring

## License

This project is open-source. See the LICENSE file for details.

## Citation

If you use this project in your research, please cite it appropriately.
