"""
W-RIZZ video traversability inference (NO TRAINING).

What it does:
  1) Automatically downloads the already-trained W-RIZZ model weights from the
     hardcoded Box link.
  2) Runs inference on the hardcoded input video name.
  3) Draws ONLY contours around the predicted traversable path/terrain.
  4) Writes a high-FPS output video.

Install dependencies first, for example:
  pip install torch torchvision opencv-python requests numpy tqdm

Put your video in the same folder as this script and name it exactly:
  input.mp4

Run:
  python wrizz_video_inference.py
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =========================
# HARD-CODED USER SETTINGS
# =========================
INPUT_VIDEO = "input.mp4"                         # <-- hardcoded video name
OUTPUT_VIDEO = "wrizz_traversable_contour.mp4"    # output video

# Official W-RIZZ trained model weights link from the W-RIZZ README.
# This is a folder shared link containing model.pth.
WRIZZ_WEIGHTS_SHARED_LINK = "https://uofi.box.com/s/s73bnggeo48o8iyzem4c5w6aduc208h2"
WEIGHTS_PATH = "weights/model.pth"

# W-RIZZ/TravNet default inference resolution used by the paper repo.
MODEL_H, MODEL_W = 240, 424

# Traversability post-processing.
# If contours miss too much traversable ground, lower THRESHOLD a little.
# If contours include too much, raise it.
THRESHOLD = 0.55
MIN_CONTOUR_AREA_FRAC = 0.004       # remove tiny contour noise; fraction of frame area
KEEP_BOTTOM_CONNECTED_ONLY = True   # keeps candidate paths touching bottom image band
BOTTOM_BAND_FRAC = 0.12             # bottom band used for path selection
DRAW_THICKNESS = 3
CONTOUR_COLOR_BGR = (0, 255, 0)

# Good-FPS settings.
USE_CUDA_IF_AVAILABLE = True
USE_FP16_ON_CUDA = True
DISPLAY_PROGRESS_EVERY_N_FRAMES = 30


# =========================
# Model architecture copied from W-RIZZ TravNetUp3NNRGB
# =========================
class TravNetUp3NNRGB(nn.Module):
    def __init__(self, pretrained: bool = False, output_size: Tuple[int, int] = (240, 424),
                 bottleneck_dim: int = 256, output_channels: int = 1):
        super().__init__()

        # torchvision compatibility: older versions use pretrained=, newer use weights=.
        try:
            model = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
        except Exception:
            model = models.resnet18(pretrained=pretrained)

        self.out_dim = (output_size[0], output_size[1])
        self.num_classes = output_channels

        # Encoder
        self.block1 = nn.Sequential(*(list(model.children())[:3]))
        self.block2 = nn.Sequential(model.maxpool, model.layer1)
        self.block3 = model.layer2
        self.block4 = model.layer3
        self.block5 = model.layer4

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.convUp1A = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.convUp1B = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.convUp2A = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.convUp2B = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.convUp3A = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.convUp3B = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.convUp4A = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.convUp4B = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.convUp5A = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.convUp5B = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(32, output_channels, kernel_size=1, stride=1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)

        x = self.bottleneck(out5)
        x = torch.cat((x, out5), dim=1)
        x = self.convUp1B(F.interpolate(self.convUp1A(x), out4.shape[2:], mode="nearest"))
        x = torch.cat((x, out4), dim=1)
        x = self.convUp2B(F.interpolate(self.convUp2A(x), out3.shape[2:], mode="nearest"))
        x = torch.cat((x, out3), dim=1)
        x = self.convUp3B(F.interpolate(self.convUp3A(x), out2.shape[2:], mode="nearest"))
        x = torch.cat((x, out2), dim=1)
        x = self.convUp4B(F.interpolate(self.convUp4A(x), out1.shape[2:], mode="nearest"))
        x = torch.cat((x, out1), dim=1)
        x = self.convUp5B(F.interpolate(self.convUp5A(x), self.out_dim, mode="nearest"))
        output = self.activation(self.final(x))
        return {"prediction": output[:, 0]}


# =========================
# Automatic Box download
# =========================
def download_file_from_box_shared_link(shared_link: str, out_path: Path) -> None:
    """Download model.pth from the public W-RIZZ Box folder link without manual steps."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        print(f"[OK] Weights already present: {out_path}")
        return

    print(f"[INFO] Downloading W-RIZZ weights from hardcoded link:\n       {shared_link}")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    # Box enterprise URLs redirect from uofi.box.com to uofi.app.box.com.
    page = session.get(shared_link, timeout=30, allow_redirects=True)
    page.raise_for_status()
    html = page.text
    base_url_match = re.match(r"(https://[^/]+)", page.url)
    if not base_url_match:
        raise RuntimeError(f"Could not parse Box base URL from {page.url}")
    base_url = base_url_match.group(1)

    shared_name_match = re.search(r'"sharedName":"([^"]+)"', html)
    request_token_match = re.search(r'"requestToken":"([^"]+)"', html)
    file_id_matches = re.findall(r'"typedID":"f_(\d+)".*?"extension":"pth".*?"name":"([^"]+\.pth)"', html)

    if not shared_name_match or not request_token_match:
        raise RuntimeError("Could not parse Box sharedName/requestToken from the weights page.")

    if file_id_matches:
        file_id, file_name = file_id_matches[0]
    else:
        # Fallback: first file in the folder.
        file_id_match = re.search(r'"typedID":"f_(\d+)"', html)
        name_match = re.search(r'"name":"([^"]+\.pth)"', html)
        if not file_id_match:
            raise RuntimeError("Could not find a .pth file id inside the Box shared folder.")
        file_id = file_id_match.group(1)
        file_name = name_match.group(1) if name_match else "model.pth"

    shared_name = shared_name_match.group(1)
    request_token = request_token_match.group(1)

    # Request a short-lived read token for the public file.
    token_url = f"{base_url}/app-api/enduserapp/elements/tokens"
    token_resp = session.post(
        token_url,
        headers={
            "Content-Type": "application/json",
            "X-Request-Token": request_token,
            "X-Box-EndUser-API": f"sharedName={shared_name}",
        },
        data=json.dumps({"fileIDs": [file_id]}),
        timeout=30,
    )
    token_resp.raise_for_status()
    read_token = token_resp.json()[file_id]["read"]

    api_headers = {
        "Authorization": f"Bearer {read_token}",
        "BoxApi": f"shared_link={shared_link}",
    }

    # Box returns a 302 to public.boxcloud.com; requests follows it and streams bytes.
    download_url = f"https://api.box.com/2.0/files/{file_id}/content"
    with session.get(download_url, headers=api_headers, timeout=60, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        downloaded = 0
        last_print = time.time()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if time.time() - last_print > 1.0:
                    if total:
                        print(f"[INFO] Downloading {file_name}: {downloaded / total * 100:5.1f}%")
                    else:
                        print(f"[INFO] Downloading {file_name}: {downloaded / 1e6:.1f} MB")
                    last_print = time.time()

    tmp_path.replace(out_path)
    print(f"[OK] Downloaded weights to: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


# =========================
# Inference helpers
# =========================
def load_wrizz_model(weights_path: Path, device: torch.device) -> nn.Module:
    model = TravNetUp3NNRGB(pretrained=False, output_size=(MODEL_H, MODEL_W), output_channels=1)

    ckpt = torch.load(str(weights_path), map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "network", "teacher", "teacher_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise RuntimeError("Downloaded checkpoint is not a PyTorch state_dict-like object.")

    # Strip common wrappers.
    clean = {}
    for k, v in ckpt.items():
        nk = k
        for prefix in ["module.", "model.", "network.", "teacher_network.", "_teacher_network."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        clean[nk] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected checkpoint keys ignored: {len(unexpected)}")
    if len(missing) > 0:
        print(f"[WARN] Missing model keys: {len(missing)}")
        print("       If this is large, the checkpoint may not match TravNetUp3NNRGB.")

    model.to(device)
    model.eval()
    if device.type == "cuda" and USE_FP16_ON_CUDA:
        model.half()
    return model


def frame_to_tensor_bchw(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
    if device.type == "cuda" and USE_FP16_ON_CUDA:
        tensor = tensor.half()
    return tensor


def traversable_contour_mask(pred_small: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Create a cleaned binary traversable region mask at original video size."""
    pred = cv2.resize(pred_small, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    mask = (pred >= THRESHOLD).astype(np.uint8) * 255

    # Smooth and clean noise.
    k = max(3, int(round(min(out_h, out_w) * 0.012)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    min_area = int(out_h * out_w * MIN_CONTOUR_AREA_FRAC)
    bottom_y = int(out_h * (1.0 - BOTTOM_BAND_FRAC))

    for idx in range(1, num):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        component = labels == idx
        if KEEP_BOTTOM_CONNECTED_ONLY and not component[bottom_y:, :].any():
            continue
        keep[component] = 255

    return keep


def draw_contours_only(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = frame_bgr.copy()
    if contours:
        cv2.drawContours(out, contours, contourIdx=-1, color=CONTOUR_COLOR_BGR, thickness=DRAW_THICKNESS, lineType=cv2.LINE_AA)
    return out


# =========================
# Main video loop
# =========================
def main() -> None:
    video_path = Path(INPUT_VIDEO)
    weights_path = Path(WEIGHTS_PATH)

    if not video_path.exists():
        print(f"[ERROR] Input video not found: {video_path.resolve()}")
        print(f"        Put your video beside this script and name it: {INPUT_VIDEO}")
        sys.exit(1)

    download_file_from_box_shared_link(WRIZZ_WEIGHTS_SHARED_LINK, weights_path)

    device = torch.device("cuda" if (USE_CUDA_IF_AVAILABLE and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    model = load_wrizz_model(weights_path, device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = src_fps if src_fps and src_fps > 1 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO}")

    print(f"[INFO] Input : {INPUT_VIDEO} ({width}x{height}, {fps:.2f} fps, {frame_count} frames)")
    print(f"[INFO] Output: {OUTPUT_VIDEO}")

    n = 0
    t0 = time.time()
    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            inp = frame_to_tensor_bchw(frame, device)
            pred = model(inp)["prediction"][0].float().detach().cpu().numpy()  # HxW in [0,1]
            mask = traversable_contour_mask(pred, height, width)
            out_frame = draw_contours_only(frame, mask)
            writer.write(out_frame)

            n += 1
            if n % DISPLAY_PROGRESS_EVERY_N_FRAMES == 0:
                elapsed = max(time.time() - t0, 1e-6)
                msg = f"[INFO] Processed {n}"
                if frame_count:
                    msg += f"/{frame_count}"
                msg += f" frames | inference/video FPS: {n / elapsed:.1f}"
                print(msg)

    cap.release()
    writer.release()
    elapsed = max(time.time() - t0, 1e-6)
    print(f"[DONE] Saved: {OUTPUT_VIDEO}")
    print(f"[DONE] Processed {n} frames in {elapsed:.2f}s ({n / elapsed:.1f} fps).")


if __name__ == "__main__":
    main()
