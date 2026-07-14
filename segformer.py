#!/usr/bin/env python3
"""
ip.py — Real-time Traversable Path Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model   : SegFormer-B0 fine-tuned on Cityscapes
          (nvidia/segformer-b0-finetuned-cityscapes-512-1024)
          76.2 mIoU · 3.7 M params · smallest/fastest SegFormer variant
          NOT TwinLite, NOT TravelNet
Training: pretrained weights downloaded automatically from HuggingFace
Input   : video.mp4  ← hardcoded here
Output  : live window — contour overlay only, nothing saved to disk
Classes : road (0) · sidewalk (1) · terrain (9)  →  covers city + off-road
Speed   : background inference thread → display thread stays ≥ 20 FPS on CPU
          dynamic INT8 quantisation of all Linear layers for ~2× CPU throughput
Press Q / ESC to quit.
"""

import sys
import time
import threading
import collections

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH   = "input.mp4"           # ← hardcoded video name
MODEL_NAME   = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"

# Inference resolution (smaller → faster CPU speed, lower accuracy)
# 512×256 gives a good speed/quality trade-off on CPU
INFER_W, INFER_H = 512, 256

# Cityscapes 19-class indices that are traversable
#   0 = road, 1 = sidewalk, 9 = terrain
TRAVERSABLE = {0, 1, 9}

# Visualisation
OVERLAY_ALPHA  = 0.30
OVERLAY_COLOR  = (0, 210, 80)       # BGR green fill
CONTOUR_COLOR  = (0, 255, 60)       # BGR bright green outline
HULL_COLOR     = (60, 220, 255)     # BGR yellow hull
CONTOUR_THICK  = 3
MIN_AREA_FRAC  = 0.003              # ignore blobs < 0.3 % of frame area

# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE (inference thread ↔ display thread)
# ─────────────────────────────────────────────────────────────────────────────
_lock        = threading.Lock()
_frame_queue = None      # latest BGR frame fed to inference thread
_mask_cache  = None      # latest uint8 traversable mask (full display res)
_infer_fps   = [0.0]     # single-element list so thread can mutate it
_stop        = threading.Event()

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    print(f"[ip.py] Downloading / loading  {MODEL_NAME}  …")
    proc  = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    net   = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    net.eval()

    # Dynamic INT8 quantisation — quantises all nn.Linear layers on CPU.
    # Gives ~1.5-2× throughput with negligible accuracy loss on Cityscapes.
    net = torch.quantization.quantize_dynamic(
        net, {torch.nn.Linear}, dtype=torch.qint8
    )
    print("[ip.py] Model ready (INT8 dynamic quantisation applied).\n")
    return proc, net

PROCESSOR, MODEL = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def _infer_frame(bgr: np.ndarray) -> np.ndarray:
    """
    Run SegFormer-B0 on one BGR frame.
    Returns a uint8 binary mask (255 = traversable) at the original frame size.
    """
    orig_h, orig_w = bgr.shape[:2]

    # Downscale for inference
    small = cv2.resize(bgr, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        inputs  = PROCESSOR(images=rgb, return_tensors="pt",
                            size={"height": INFER_H, "width": INFER_W})
        outputs = MODEL(**inputs)
        # logits: (1, 19, H/4, W/4)  →  upsample to infer size
        logits  = outputs.logits
        logits  = F.interpolate(
            logits,
            size=(INFER_H, INFER_W),
            mode="bilinear",
            align_corners=False,
        )
        pred = logits.argmax(dim=1)[0].byte().numpy()   # (INFER_H, INFER_W) uint8

    # Build binary traversable mask
    trav = np.zeros_like(pred, dtype=np.uint8)
    for cls in TRAVERSABLE:
        trav[pred == cls] = 255

    # Morphological polish: fill holes, remove noise
    ker   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    trav  = cv2.morphologyEx(trav, cv2.MORPH_CLOSE, ker, iterations=2)
    trav  = cv2.morphologyEx(trav, cv2.MORPH_OPEN,  ker, iterations=1)

    # Scale back to original display resolution
    return cv2.resize(trav, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def _inference_thread():
    """Daemon thread: pulls latest frame, runs inference, stores mask."""
    global _mask_cache
    while not _stop.is_set():
        with _lock:
            frame = _frame_queue
        if frame is None:
            time.sleep(0.005)
            continue
        t0 = time.perf_counter()
        try:
            mask = _infer_frame(frame)
            dt   = time.perf_counter() - t0
            with _lock:
                _mask_cache   = mask
                _infer_fps[0] = 1.0 / dt if dt > 0 else 0.0
        except Exception as e:
            print(f"[inference] {e}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def _overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw filled semi-transparent region + contour outline on frame."""
    h, w  = frame.shape[:2]
    min_px = int(MIN_AREA_FRAC * h * w)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid   = [c for c in cnts if cv2.contourArea(c) >= min_px]
    if not valid:
        return frame

    valid.sort(key=cv2.contourArea, reverse=True)
    valid = valid[:8]     # keep up to 8 largest regions

    # Semi-transparent fill
    tmp = frame.copy()
    cv2.fillPoly(tmp, valid, OVERLAY_COLOR)
    cv2.addWeighted(tmp, OVERLAY_ALPHA, frame, 1.0 - OVERLAY_ALPHA, 0, frame)

    # Contour outline
    cv2.drawContours(frame, valid, -1, CONTOUR_COLOR, CONTOUR_THICK, cv2.LINE_AA)

    # Convex hull of all traversable points (shows overall "safe zone" boundary)
    all_pts = np.vstack(valid)
    hull    = cv2.convexHull(all_pts)
    cv2.polylines(frame, [hull], isClosed=True,
                  color=HULL_COLOR, thickness=2, lineType=cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global _frame_queue, _mask_cache

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sys.exit(f"[ip.py] ERROR: cannot open '{VIDEO_PATH}'")

    vid_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[ip.py] Video : {VIDEO_PATH}  |  {vid_w}×{vid_h}  {vid_fps:.1f} fps  {n_frames} frames")
    print("[ip.py] Press  Q / ESC  to quit.\n")

    # Start background inference thread
    thread = threading.Thread(target=_inference_thread, daemon=True, name="infer")
    thread.start()

    # Create a resizable window and set default size (e.g., 1024x768)
    # since video resolution can be extremely high (e.g., 4K)
    window_name = "Traversable Path Detection  [ip.py]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    display_fps_hist = collections.deque(maxlen=40)
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop
            continue

        # Push frame to inference thread (non-blocking)
        with _lock:
            _frame_queue = frame.copy()
            mask = _mask_cache

        # Draw traversable overlay if mask is available
        if mask is not None:
            frame = _overlay(frame, mask.copy())

        # HUD — FPS counters
        t_now = time.perf_counter()
        dt    = t_now - t_prev
        t_prev = t_now
        display_fps_hist.append(1.0 / dt if dt > 0 else 0.0)
        d_fps = sum(display_fps_hist) / len(display_fps_hist)

        with _lock:
            i_fps = _infer_fps[0]

        cv2.putText(frame, f"Display: {d_fps:5.1f} fps",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Infer : {i_fps:5.1f} fps",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "SegFormer-B0 | Cityscapes | road+sidewalk+terrain",
                    (10, vid_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 160), 1, cv2.LINE_AA)

        cv2.imshow("Traversable Path Detection  [ip.py]", frame)

        # Throttle display to not exceed video FPS (looks natural)
        delay_ms = max(1, int(1000.0 / vid_fps) - max(1, int(dt * 1000)))
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # Q or ESC
            break

    _stop.set()
    cap.release()
    cv2.destroyAllWindows()
    print("[ip.py] Exited cleanly.")


if __name__ == "__main__":
    main()
