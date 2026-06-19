import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)

        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        grads = self.gradients
        activations = self.activations

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)

        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()[0]

    def preprocess_image(img):
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img, (224, 224))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        tensor = transform(Image.fromarray(img_resized)).unsqueeze(0)

        return tensor, img_resized

    def overlay_cam_on_image(img, cam):
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        heatmap = (cam * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(
            img,
            0.6,
            heatmap,
            0.4,
            0
        )

        return overlay
# Notice that following codes are wrote for separated test of each part

# EfficientNet Grad-Cam Runner
'''def run_gradcam_efficientnet(model, image_path, device, save_path="cam.jpg"):
    model.eval()
    model.to(device)

    input_tensor, img = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    target_layer = model.features[-1]

    cam_extractor = GradCAM(model, target_layer)

    cam = cam_extractor.generate_cam(input_tensor)

    overlay = overlay_cam_on_image(img, cam)

    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return cam, overlay'''

#CNN-LSTM Temporal Grad-Cam
'''def extract_frames(video_path, frame_count=8, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)

    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // frame_count, 1)

    idx = 0
    saved = 0

    while cap.isOpened() and saved < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, size)
            frames.append(frame)
            saved += 1

        idx += 1

    cap.release()

    return np.array(frames) # (T,H,W,C)

#Temporal input builder
def build_temporal_tensor(frames):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor_list = []

    for f in frames:
        t = transform(Image.fromarray(f))
        tensor_list.append(t)

    return torch.stack(tensor_list, dim=0).unsqueeze(0)  # (1,T,C,H,W)

#CNN-LSTM Grad-CAM(frame-level approximation)
def run_gradcam_temporal(model, video_path, device, save_dir="temporal_cam"):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    model.to(device)

    frames = extract_frames(video_path)
    input_tensor = build_temporal_tensor(frames).to(device)

    cnn = model.feature_extractor[0]
    target_layer = cnn[-1]

    cam_extractor = GradCAM(cnn, target_layer)

    cams = []

    for t in range(frames.shape[0]):
        frame = frames[t]

        tensor = input_tensor[:, t]  # (1,C,H,W)

        cam = cam_extractor.generate_cam(tensor)

        overlay = overlay_cam_on_image(frame, cam)

        save_path = os.path.join(save_dir, f"frame_{t}.jpg")

        cv2.imwrite(
            save_path,
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

        cams.append(cam)

    return cams

#Batch inference + worst-case samples+ full pipeline utilities
def predict(model, tensor, device):
    model.eval()
    model.to(device)

    with torch.no_grad():
        out = model(tensor.to(device))
        prob = F.softmax(out, dim=1)
        pred = torch.argmax(prob, dim=1)

    return pred.cpu().numpy()[0], prob.cpu().numpy()[0]

#Batch Grad-CAM(dataset level)
def run_batch_gradcam(model, image_paths, device, save_dir="batch_cam"):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    model.to(device)

    target_layer = model.features[-1]
    cam_extractor = GradCAM(model, target_layer)

    results = []

    for i, path in enumerate(image_paths):

        tensor, img = preprocess_image(path)
        tensor = tensor.to(device)

        cam = cam_extractor.generate_cam(tensor)
        overlay = overlay_cam_on_image(img, cam)

        pred, prob = predict(model, tensor, device)

        save_path = os.path.join(save_dir, f"{i}_cam.jpg")

        cv2.imwrite(
            save_path,
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

        results.append({
            "path": path,
            "pred": int(pred),
            "confidence": float(np.max(prob))
        })

    return results

#Worst-case(important for thesis discussion)
def visualize_worst_cases(model, dataset_paths, device, save_dir="worst_cases"):
    os.makedirs(save_dir, exist_ok=True)

    records = []

    for path in dataset_paths:

        tensor, img = preprocess_image(path)
        tensor = tensor.to(device)

        with torch.no_grad():
            out = model(tensor)
            prob = F.softmax(out, dim=1).cpu().numpy()[0]

        pred = np.argmax(prob)
        conf = np.max(prob)

        records.append((path, pred, conf))

    # sort lowest confidence
    records = sorted(records, key=lambda x: x[2])[:20]

    target_layer = model.features[-1]
    cam_extractor = GradCAM(model, target_layer)

    for i, (path, pred, conf) in enumerate(records):

        tensor, img = preprocess_image(path)
        tensor = tensor.to(device)

        cam = cam_extractor.generate_cam(tensor)

        overlay = overlay_cam_on_image(img, cam)

        save_path = os.path.join(save_dir, f"worst_{i}_p{pred}_c{conf:.2f}.jpg")

        cv2.imwrite(
            save_path,
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

#FULL PIPELINE WRAPPER
def run_full_gradcam_pipeline(model, image_list, video_list, device):
    print("[1] Running image Grad-CAM...")
    run_batch_gradcam(model, image_list, device)

    print("[2] Running video Grad-CAM...")
    for v in video_list:
        run_gradcam_temporal(model, v, device)

    print("[3] Extracting worst cases...")
    visualize_worst_cases(model, image_list, device)

    print("[DONE] All Grad-CAM outputs saved.")
'''