import argparse
import os
import platform
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn

IMG_SIZE = 128
CLASSES = ["A", "B", "C", "D", "E"]
CHANNEL_MEAN = (0.485, 0.456, 0.406)
CHANNEL_STD = (0.229, 0.224, 0.225)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        scale = self.pool(x).view(batch_size, channels)
        scale = self.fc(scale).view(batch_size, channels, 1, 1)
        return x * scale


class ResidualASLBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.act = nn.GELU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        x = x + residual
        return self.act(x)


def build_model(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
        nn.BatchNorm2d(32),
        nn.GELU(),
        ResidualASLBlock(32, 64, stride=2),
        ResidualASLBlock(64, 64),
        nn.Dropout2d(0.08),
        ResidualASLBlock(64, 128, stride=2),
        ResidualASLBlock(128, 128),
        nn.Dropout2d(0.12),
        ResidualASLBlock(128, 192, stride=2),
        ResidualASLBlock(192, 192),
        nn.Dropout2d(0.16),
        ResidualASLBlock(192, 256, stride=2),
        ResidualASLBlock(256, 256),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.LayerNorm(256),
        nn.Linear(256, 192),
        nn.GELU(),
        nn.Dropout(0.35),
        nn.Linear(192, num_classes),
    )


def load_model(weights_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model file not found: {weights_path}")

    model = build_model(len(CLASSES)).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_frame(frame_bgr, device: torch.device) -> torch.Tensor:
    import cv2

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(
        frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
    )
    tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(CHANNEL_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(CHANNEL_STD, dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device, non_blocking=True)


def predict_topk(
    model: nn.Module, input_tensor: torch.Tensor, k: int = 3
) -> List[Tuple[str, float]]:
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(k, len(CLASSES)))
    return [
        (CLASSES[idx.item()], float(prob.item()))
        for prob, idx in zip(top_probs, top_indices)
    ]


def _largest_bbox_from_mask(mask, min_area_px: float) -> Optional[Tuple[int, int, int, int]]:
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_px:
        return None
    return cv2.boundingRect(largest)


def detect_roi_bbox(frame_bgr) -> Optional[Tuple[int, int, int, int]]:
    import cv2

    h, w = frame_bgr.shape[:2]
    min_area_px = max(1200.0, 0.01 * h * w)

    frame_blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)

    # Skin-color based mask (good for bare-hand ASL input).
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2YCrCb)
    skin_hsv = cv2.inRange(hsv, (0, 30, 50), (25, 180, 255))
    skin_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin_mask = cv2.bitwise_and(skin_hsv, skin_ycrcb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    bbox = _largest_bbox_from_mask(skin_mask, min_area_px=min_area_px)
    if bbox is not None:
        return bbox

    # Fallback for non-skin inputs (e.g., gloves): use adaptive foreground mask.
    gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _largest_bbox_from_mask(fg_mask, min_area_px=min_area_px)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live webcam/video test for the ASL CNN model."
    )
    parser.add_argument("model_path", help="Path to trained model weights (.pth/.pt).")
    parser.add_argument(
        "--source",
        default="auto",
        help="Camera index (e.g. 0), 'auto', or video file path.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "avfoundation", "any"],
        default="auto",
        help="OpenCV backend preference for camera capture.",
    )
    parser.add_argument(
        "--scan-max",
        type=int,
        default=10,
        help="Maximum camera index to probe for auto/list modes.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List detected cameras and exit.",
    )
    return parser.parse_args()


def parse_source(source_arg: str):
    source_text = source_arg.strip()
    if source_text.lower() == "auto":
        return "auto"
    if source_text.isdigit():
        return int(source_text)

    file_path = Path(source_text).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(
            f"Video source path not found: {file_path}. Use a camera index (e.g. 0) "
            "or a valid video file path."
        )
    return str(file_path)


def select_backends(cv2, backend_arg: str):
    if backend_arg == "avfoundation":
        return [cv2.CAP_AVFOUNDATION]
    if backend_arg == "any":
        return [cv2.CAP_ANY]

    if platform.system() == "Darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def backend_name(cv2, backend_id: int) -> str:
    names = {
        cv2.CAP_ANY: "CAP_ANY",
        cv2.CAP_AVFOUNDATION: "CAP_AVFOUNDATION",
    }
    return names.get(backend_id, str(backend_id))


def configure_capture(cv2, cap, is_camera_source: bool) -> None:
    # Keep latency low for live camera feeds.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if is_camera_source:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)


def warmup_capture(cap, retries: int = 60, delay_s: float = 0.03):
    for _ in range(retries):
        success, frame = cap.read()
        if success and frame is not None and getattr(frame, "size", 0) > 0:
            return True, frame
        time.sleep(delay_s)
    return False, None


def open_camera_index(cv2, camera_index: int, backends):
    attempts = []
    for backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        attempts.append(
            f"camera index {camera_index} with {backend_name(cv2, backend)}"
        )
        if cap.isOpened():
            configure_capture(cv2, cap, is_camera_source=True)
            has_frame, _ = warmup_capture(cap, retries=60)
            if has_frame:
                return cap, attempts
        cap.release()
    return None, attempts


def list_cameras(cv2, backends, max_index: int):
    cameras = []
    attempts = []
    for camera_index in range(max_index + 1):
        cap, camera_attempts = open_camera_index(cv2, camera_index, backends)
        attempts.extend(camera_attempts)
        if cap is not None:
            cameras.append((camera_index, capture_debug_info(cap)))
            cap.release()
    return cameras, attempts


def open_capture(cv2, source, backends, scan_max: int):
    attempts = []

    if isinstance(source, str):
        if source == "auto":
            for camera_index in range(scan_max + 1):
                cap, camera_attempts = open_camera_index(cv2, camera_index, backends)
                attempts.extend(camera_attempts)
                if cap is not None:
                    return cap, camera_index, attempts
            return None, None, attempts

        cap = cv2.VideoCapture(source)
        attempts.append(f"source='{source}' with default backend")
        if cap.isOpened():
            configure_capture(cv2, cap, is_camera_source=False)
            has_frame, _ = warmup_capture(cap, retries=30)
            if has_frame:
                return cap, source, attempts
        cap.release()
        return None, source, attempts

    cap, camera_attempts = open_camera_index(cv2, source, backends)
    attempts.extend(camera_attempts)
    if cap is not None:
        return cap, source, attempts
    return None, source, attempts


def recover_frame(cap, retries: int = 15, delay_s: float = 0.02):
    for _ in range(retries):
        success, frame = cap.read()
        if success and frame is not None and getattr(frame, "size", 0) > 0:
            return True, frame
        time.sleep(delay_s)
    return False, None


def macos_camera_help() -> str:
    if platform.system() != "Darwin":
        return ""

    term_program = os.environ.get("TERM_PROGRAM", "your terminal app")
    return (
        "\nmacOS camera access is likely blocked for this terminal.\n"
        "1) Open System Settings -> Privacy & Security -> Camera and allow camera access for "
        f"{term_program}.\n"
        "2) If it still fails, reset permission and rerun from the same terminal:\n"
        "   tccutil reset Camera\n"
    )


def capture_debug_info(cap) -> str:
    backend = int(cap.get(42))  # cv2.CAP_PROP_BACKEND
    width = int(cap.get(3))  # cv2.CAP_PROP_FRAME_WIDTH
    height = int(cap.get(4))  # cv2.CAP_PROP_FRAME_HEIGHT
    fps = cap.get(5)  # cv2.CAP_PROP_FPS
    return f"backend={backend}, resolution={width}x{height}, fps={fps:.2f}"


def main():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for webcam testing. Install it with: "
            "pip install opencv-python"
        ) from exc

    args = parse_args()
    source = parse_source(args.source)
    if args.scan_max < 0:
        raise ValueError("--scan-max must be >= 0")

    backends = select_backends(cv2, args.backend)
    if args.list_cameras:
        cameras, _ = list_cameras(cv2, backends, args.scan_max)
        if not cameras:
            raise RuntimeError(
                "No working cameras detected.\n"
                f"Scanned indices 0..{args.scan_max}.\n"
                f"{macos_camera_help()}"
            )
        print("Detected cameras:")
        for camera_index, info in cameras:
            print(f"- index {camera_index}: {info}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    cap, selected_source, attempts = open_capture(cv2, source, backends, args.scan_max)

    if cap is None:
        attempts_text = "\n".join(f"- {item}" for item in attempts)
        raise RuntimeError(
            "Could not open video input. Tried:\n"
            f"{attempts_text}"
            f"{macos_camera_help()}"
        )

    print(f"Using device: {device}")
    print(f"Loaded model: {args.model_path}")
    print(f"Video source: {args.source} (selected: {selected_source})")
    print("Press 'q' to quit.")
    print(f"Capture info: {capture_debug_info(cap)}")

    try:
        cv2.namedWindow("ASL Live Test", cv2.WINDOW_NORMAL)

        while True:
            success, frame = cap.read()
            if not success:
                success, frame = recover_frame(cap)
            if not success:
                raise RuntimeError(
                    "Failed to read frame from video source.\n"
                    f"Capture info: {capture_debug_info(cap)}\n"
                    "Try --source auto, a different source index (e.g. --source 1), "
                    "close other camera apps, or force backend with --backend avfoundation/any."
                    f"{macos_camera_help()}"
                )

            bbox = detect_roi_bbox(frame)
            if bbox is not None:
                x, y, w, h = bbox
                roi = frame[y : y + h, x : x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 255), 2)
            else:
                roi = frame

            input_tensor = preprocess_frame(roi, device)
            top3 = predict_topk(model, input_tensor, k=3)

            text_y = 30
            for rank, (label, confidence) in enumerate(top3, start=1):
                overlay_text = f"#{rank}: {label} ({confidence * 100:.1f}%)"
                cv2.putText(
                    frame,
                    overlay_text,
                    (20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                text_y += 30
            cv2.imshow("ASL Live Test", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
