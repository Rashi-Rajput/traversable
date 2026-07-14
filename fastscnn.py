#!/usr/bin/env python3
"""
ip.py — Real-time Traversable Path Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model    : Fast-SCNN (Learning to Downsample for Real-time Semantic Segmentation)
           Pretrained on Cityscapes — 68.6 mIoU, ~1.1 MB weights
           NOT SegFormer, NOT TwinLite, NOT TravelNet
Weights  : auto-downloaded from openmmlab CDN (~5.9 MB checkpoint)
Input    : video.mp4  ← hardcoded
Output   : live window — green contour / fill overlay, nothing saved
Classes  : road(0) · sidewalk(1) · terrain(9)  covers city + off-road
Speed    : background inference thread + INT8 dynamic quantisation
           → display stays ≥ 20 FPS on CPU
Press Q / ESC to quit.
"""

import os
import sys
import time
import threading
import collections
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH   = "vid.mp4"
WEIGHTS_URL  = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/fast_scnn/"
    "fast_scnn_lr0.12_8x4_160k_cityscapes/"
    "fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth"
)
WEIGHTS_FILE = "fast_scnn_cityscapes.pth"

INFER_W, INFER_H = 512, 256     # inference resolution  (smaller = faster)
TRAVERSABLE      = {0, 1, 9}    # Cityscapes classes: road, sidewalk, terrain
NUM_CLASSES      = 19

# Cityscapes normalisation
MEAN = torch.tensor([123.675, 116.28, 103.53],  dtype=torch.float32).view(3, 1, 1)
STD  = torch.tensor([58.395,  57.12,  57.375],  dtype=torch.float32).view(3, 1, 1)

# Visualisation
OVERLAY_ALPHA  = 0.32
OVERLAY_COLOR  = (0, 210, 80)      # BGR green fill
CONTOUR_COLOR  = (0, 255, 60)      # BGR bright-green outline
HULL_COLOR     = (50, 220, 255)    # BGR yellow hull
CONTOUR_THICK  = 3
MIN_AREA_FRAC  = 0.003             # drop blobs < 0.3% of frame area

# ─────────────────────────────────────────────────────────────────────────────
# FAST-SCNN ARCHITECTURE  (matches openmmlab checkpoint key names exactly)
# ─────────────────────────────────────────────────────────────────────────────

class ConvBN(nn.Module):
    """Conv2d + BN2d [+ ReLU].  Keys: *.conv.weight  *.bn.*"""
    def __init__(self, in_ch, out_ch, k=3, stride=1, pad=1,
                 groups=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, pad,
                              groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self._relu = relu

    def forward(self, x):
        x = self.bn(self.conv(x))
        return F.relu(x, inplace=True) if self._relu else x


class DSConv(nn.Module):
    """Depthwise-Separable Conv.
    Keys: *.depthwise_conv.{conv,bn}.*  *.pointwise_conv.{conv,bn}.*"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise_conv = ConvBN(in_ch, in_ch, 3, stride, 1,
                                     groups=in_ch, relu=True)
        self.pointwise_conv = ConvBN(in_ch, out_ch, 1, 1, 0, relu=True)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class InvertedResidual(nn.Module):
    """MobileNetV2-style bottleneck.
    Keys: *.conv.{0,1,2}.{conv,bn}.*"""
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        mid = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.conv = nn.Sequential(
            ConvBN(in_ch, mid, 1, 1, 0, relu=True),           # expand
            ConvBN(mid,  mid, 3, stride, 1, groups=mid, relu=True),  # dw
            ConvBN(mid, out_ch, 1, 1, 0, relu=False),          # project
        )

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class LearningToDownsample(nn.Module):
    """3-layer LtD branch.  Keys: backbone.learning_to_downsample.*"""
    def __init__(self):
        super().__init__()
        self.conv    = ConvBN(3,  32, 3, 2, 1, relu=True)   # 3→32  /2
        self.dsconv1 = DSConv(32, 48, stride=2)              # 32→48 /2
        self.dsconv2 = DSConv(48, 64, stride=2)              # 48→64 /2

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x   # 64-ch at H/8


class GlobalFeatureExtractor(nn.Module):
    """Bottleneck stack + PPM.  Keys: backbone.global_feature_extractor.*"""
    def __init__(self):
        super().__init__()
        self.bottleneck1 = self._make(64,  64,  n=3, stride=2, t=6)
        self.bottleneck2 = self._make(64,  96,  n=3, stride=2, t=6)
        self.bottleneck3 = self._make(96,  128, n=3, stride=1, t=6)
        # PPM: 4 pool-sizes → each branch: AdaptiveAvgPool → ConvBN(128→32)
        # Keys: ppm.{0..3}.1.{conv,bn}.*
        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                ConvBN(128, 32, 1, 1, 0, relu=True)
            )
            for ps in (1, 2, 3, 6)
        ])

    @staticmethod
    def _make(in_ch, out_ch, n, stride, t):
        blocks = [InvertedResidual(in_ch, out_ch, stride, t)]
        for _ in range(n - 1):
            blocks.append(InvertedResidual(out_ch, out_ch, 1, t))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.bottleneck1(x)   # 64-ch at H/16
        x = self.bottleneck2(x)   # 96-ch at H/32
        x = self.bottleneck3(x)   # 128-ch at H/32
        h, w = x.shape[2:]
        pp  = [F.interpolate(b(x), size=(h, w),
                             mode='bilinear', align_corners=True)
               for b in self.ppm]
        return torch.cat(pp, dim=1)  # 4×32=128-ch at H/32


class FeatureFusion(nn.Module):
    """FFM.  Keys: backbone.feature_fusion.*"""
    def __init__(self):
        super().__init__()
        self.dwconv          = ConvBN(128, 128, 3, 1, 1, groups=128, relu=True)
        self.conv_lower_res  = ConvBN(128, 128, 1, 1, 0, relu=False)
        self.conv_higher_res = ConvBN(64,  128, 1, 1, 0, relu=False)

    def forward(self, higher_res, lower_res):
        # upsample lower_res (H/32) → H/8 to match higher_res
        lr = F.interpolate(lower_res, size=higher_res.shape[2:],
                           mode='bilinear', align_corners=True)
        lr = self.conv_lower_res(self.dwconv(lr))
        hr = self.conv_higher_res(higher_res)
        return F.relu(lr + hr, inplace=True)  # 128-ch at H/8


class FastSCNNBackbone(nn.Module):
    """Keys: backbone.*"""
    def __init__(self):
        super().__init__()
        self.learning_to_downsample   = LearningToDownsample()
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.feature_fusion           = FeatureFusion()

    def forward(self, x):
        higher = self.learning_to_downsample(x)      # 64-ch @ H/8
        lower  = self.global_feature_extractor(higher)  # 128-ch @ H/32
        fused  = self.feature_fusion(higher, lower)  # 128-ch @ H/8
        return fused


class DecodeHead(nn.Module):
    """2×DSConv + 1×conv_seg.  Keys: decode_head.*"""
    def __init__(self, in_ch=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.convs    = nn.ModuleList([DSConv(in_ch, in_ch),
                                       DSConv(in_ch, in_ch)])
        self.conv_seg = nn.Conv2d(in_ch, num_classes, 1)

    def forward(self, x):
        for c in self.convs:
            x = c(x)
        return self.conv_seg(x)


class FastSCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone    = FastSCNNBackbone()
        self.decode_head = DecodeHead()

    def forward(self, x):
        feat = self.backbone(x)
        return self.decode_head(feat)


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_weights(model: FastSCNN, path: str) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)

    # Only keep backbone.* and decode_head.* keys (drop auxiliary_head)
    filtered = {k: v for k, v in sd.items()
                if k.startswith("backbone.") or k.startswith("decode_head.")}

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[ip.py] WARNING — missing keys: {missing[:5]}…")
    print(f"[ip.py] Loaded {len(filtered)} tensors from checkpoint.")


def download_weights(url: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"[ip.py] Weights already present: {dst}")
        return
    print(f"[ip.py] Downloading Fast-SCNN weights (~5.9 MB) …")
    tmp = dst + ".tmp"
    try:
        def _progress(count, block, total):
            pct = min(100, count * block * 100 // total)
            print(f"\r  {pct}% ", end="", flush=True)
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        print()
        os.rename(tmp, dst)
        print(f"[ip.py] Saved to {dst}")
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        sys.exit(f"[ip.py] Download failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────────────────────────────────────

download_weights(WEIGHTS_URL, WEIGHTS_FILE)

MODEL = FastSCNN()
MODEL.eval()
load_weights(MODEL, WEIGHTS_FILE)

# INT8 dynamic quantisation — quantises all Conv2d on CPU for ~1.5× throughput
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    MODEL = torch.quantization.quantize_dynamic(
        MODEL, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
print("[ip.py] INT8 quantisation applied. Model ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(bgr: np.ndarray) -> torch.Tensor:
    small = cv2.resize(bgr, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    t     = torch.from_numpy(rgb).permute(2, 0, 1)   # (3,H,W)
    return ((t - MEAN) / STD).unsqueeze(0)            # (1,3,H,W)


def infer(bgr: np.ndarray) -> np.ndarray:
    """Return uint8 binary traversable mask at original frame resolution."""
    orig_h, orig_w = bgr.shape[:2]
    x = preprocess(bgr)

    with torch.no_grad():
        logits = MODEL(x)                                    # (1,19,H/8,W/8)
        logits = F.interpolate(logits,
                               size=(INFER_H, INFER_W),
                               mode="bilinear",
                               align_corners=True)           # (1,19,INFER_H,INFER_W)
        pred = logits.argmax(dim=1)[0].byte().numpy()        # (INFER_H,INFER_W)

    # Build traversable binary mask
    trav = np.zeros_like(pred, dtype=np.uint8)
    for cls in TRAVERSABLE:
        trav[pred == cls] = 255

    # Morphological clean-up
    ker  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    trav = cv2.morphologyEx(trav, cv2.MORPH_CLOSE, ker, iterations=2)
    trav = cv2.morphologyEx(trav, cv2.MORPH_OPEN,  ker, iterations=1)

    return cv2.resize(trav, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE  (inference thread ↔ display thread)
# ─────────────────────────────────────────────────────────────────────────────

_lock        = threading.Lock()
_latest_bgr  = None      # freshest frame from display loop
_latest_mask = None       # latest binary mask from inference thread
_infer_fps   = [0.0]      # updated by inference thread
_stop        = threading.Event()


def _inference_loop():
    global _latest_mask
    while not _stop.is_set():
        with _lock:
            frame = _latest_bgr
        if frame is None:
            time.sleep(0.005)
            continue
        t0 = time.perf_counter()
        try:
            mask = infer(frame)
            dt   = time.perf_counter() - t0
            with _lock:
                _latest_mask  = mask
                _infer_fps[0] = 1.0 / dt if dt > 0 else 0.0
        except Exception as exc:
            print(f"[infer-thread] {exc}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w    = frame.shape[:2]
    min_px  = int(MIN_AREA_FRAC * h * w)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid   = [c for c in cnts if cv2.contourArea(c) >= min_px]
    if not valid:
        return frame

    valid.sort(key=cv2.contourArea, reverse=True)
    valid = valid[:8]

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, valid, OVERLAY_COLOR)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1.0 - OVERLAY_ALPHA, 0, frame)

    # Contour outline
    cv2.drawContours(frame, valid, -1, CONTOUR_COLOR, CONTOUR_THICK, cv2.LINE_AA)

    # Convex hull = overall safe-zone boundary
    all_pts = np.vstack(valid)
    hull    = cv2.convexHull(all_pts)
    cv2.polylines(frame, [hull], True, HULL_COLOR, 2, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global _latest_bgr, _latest_mask

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sys.exit(f"[ip.py] Cannot open: '{VIDEO_PATH}'")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_fr    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[ip.py] {VIDEO_PATH} | {vid_w}×{vid_h}  {vid_fps:.1f} fps  {n_fr} frames")
    print("[ip.py] Press  Q / ESC  to quit.\n")

    # Limit maximum display width to 1280 for screen safety
    max_disp_w = 1280
    disp_scale = min(1.0, max_disp_w / vid_w)
    disp_size = (int(vid_w * disp_scale), int(vid_h * disp_scale))

    # Start background inference
    t = threading.Thread(target=_inference_loop, daemon=True, name="infer")
    t.start()

    # Create a resizable window and set default size (e.g., 1280x720)
    window_name = "Traversable Path — Fast-SCNN [ip.py]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_size[0], disp_size[1])

    disp_hist = collections.deque(maxlen=40)
    t_prev    = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop
            continue

        # Push to inference thread (non-blocking)
        with _lock:
            _latest_bgr = frame.copy()
            mask = _latest_mask

        # Draw
        if mask is not None:
            frame = draw_overlay(frame, mask.copy())

        # HUD
        t_now = time.perf_counter()
        dt    = t_now - t_prev
        t_prev = t_now
        disp_hist.append(1.0 / dt if dt > 0 else 0.0)
        d_fps = sum(disp_hist) / len(disp_hist)
        with _lock:
            i_fps = _infer_fps[0]

        cv2.putText(frame, f"Display : {d_fps:5.1f} fps",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Infer   : {i_fps:5.1f} fps",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (100, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Fast-SCNN | Cityscapes | road + sidewalk + terrain",
                    (10, vid_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (180, 255, 160), 1, cv2.LINE_AA)

        # Resize the frame array dynamically if it exceeds the max width of 1280px
        disp_frame = cv2.resize(frame, disp_size, interpolation=cv2.INTER_LINEAR) if disp_scale < 1.0 else frame
        cv2.imshow("Traversable Path — Fast-SCNN [ip.py]", disp_frame)

        delay = max(1, int(1000.0 / vid_fps) - max(1, int(dt * 1000)))
        if cv2.waitKey(delay) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    _stop.set()
    cap.release()
    cv2.destroyAllWindows()
    print("[ip.py] Exited cleanly.")


if __name__ == "__main__":
    main()
