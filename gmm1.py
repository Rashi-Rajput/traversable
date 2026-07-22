# Save as traversability_accurate.py
import os
import argparse
import time
import cv2
import numpy as np
from collections import deque

# ----------------------------- CONFIG -----------------------------
DEFAULT_VIDEO = "input.mp4"
MAX_PROC_WIDTH = 640          # Downscale for CPU speed
SPATIAL_SIGMA_X = 0.35        # Spatial prior spread (width fraction)
SPATIAL_SIGMA_Y = 0.25        # Spatial prior spread (height fraction)
SPATIAL_CENTER_Y = 0.82       # Prior peak height ratio

COLOR_WEIGHT = 0.45
TEXTURE_WEIGHT = 0.30
SPATIAL_WEIGHT = 0.25

TEMPORAL_ALPHA = 0.65
MOTION_GATE_THRESH = 18       # Frame diff threshold to pause temporal smoothing
ADAPTIVE_UPDATE_RATE = 0.08   # Online model learning rate
CONFIDENCE_UPDATE_THRESH = 0.65

MORPH_KERNEL = 5
MIN_AREA_RATIO = 0.015        # Min component area relative to frame

# ============================================================
# SPATIAL PRIOR (Soft bottom-center bias, works indoor/outdoor)
# ============================================================
def build_spatial_prior(h, w):
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h * SPATIAL_CENTER_Y
    sx, sy = w * SPATIAL_SIGMA_X, h * SPATIAL_SIGMA_Y
    prior = np.exp(-((x - cx)**2 / (2 * sx**2) + (y - cy)**2 / (2 * sy**2)))
    return prior.astype(np.float32)

# ============================================================
# ONLINE TRAVERSABLE MODEL (LAB space, diagonal covariance)
# ============================================================
class OnlineTraversableModel:
    def __init__(self, update_rate=ADAPTIVE_UPDATE_RATE):
        self.mean = None
        self.var = None
        self.update_rate = update_rate
        self.initialized = False

    def _init_from_seed(self, lab_img, seed_mask):
        pts = lab_img[seed_mask > 0]
        if len(pts) < 50:
            return False
        self.mean = pts.mean(axis=0).astype(np.float32)
        self.var = pts.var(axis=0).astype(np.float32) + 1e-3
        self.initialized = True
        return True

    def update(self, lab_img, conf_mask):
        pts = lab_img[conf_mask > 0]
        if len(pts) < 30:
            return
        new_mean = pts.mean(axis=0).astype(np.float32)
        new_var = pts.var(axis=0).astype(np.float32) + 1e-3
        self.mean = (1 - self.update_rate) * self.mean + self.update_rate * new_mean
        self.var = (1 - self.update_rate) * self.var + self.update_rate * new_var

    def score(self, lab_img):
        if not self.initialized:
            return np.zeros(lab_img.shape[:2], dtype=np.float32)
        # Mahalanobis-like distance with diagonal covariance
        diff = lab_img.astype(np.float32) - self.mean
        dist = np.sum((diff ** 2) / self.var, axis=2)
        # Convert to probability-like score (0-1)
        score = np.exp(-dist / 2.0)
        return np.clip(score, 0, 1)

# ============================================================
# TEXTURE SMOOTHNESS CUE (Low local variance = likely traversable)
# ============================================================
def texture_smoothness_score(gray):
    gray_f = gray.astype(np.float32) / 255.0
    mu = cv2.GaussianBlur(gray_f, (15, 15), 0)
    mu2 = cv2.GaussianBlur(gray_f ** 2, (15, 15), 0)
    var = np.clip(mu2 - mu ** 2, 0, 1)
    # Invert: smooth areas -> high score
    score = 1.0 - np.sqrt(var)
    return np.clip(score, 0, 1)

# ============================================================
# TEMPORAL SMOOTHER (EMA + motion gating, CPU-friendly)
# ============================================================
class TemporalSmoother:
    def __init__(self, alpha=TEMPORAL_ALPHA, motion_thr=MOTION_GATE_THRESH):
        self.alpha = alpha
        self.motion_thr = motion_thr
        self.prev_score = None
        self.prev_gray = None

    def update(self, gray, curr_score):
        if self.prev_score is None:
            self.prev_score = curr_score.copy()
            self.prev_gray = gray.copy()
            return curr_score.copy()

        # Frame difference for motion gating
        frame_diff = cv2.absdiff(self.prev_gray, gray)
        frame_diff = cv2.GaussianBlur(frame_diff, (5, 5), 0)
        motion_mask = (frame_diff < self.motion_thr).astype(np.float32)

        # EMA with motion-aware blending
        smooth = self.alpha * curr_score + (1 - self.alpha) * self.prev_score
        # Where motion is high, trust current frame more
        out = motion_mask * smooth + (1 - motion_mask) * curr_score
        out = np.clip(out, 0, 1)

        self.prev_score = out.copy()
        self.prev_gray = gray.copy()
        return out

    def reset(self):
        self.prev_score = None
        self.prev_gray = None

# ============================================================
# SCENE CUT DETECTOR (Histogram + frame diff spike)
# ============================================================
class SceneCutDetector:
    def __init__(self, thr=0.45):
        self.thr = thr
        self.prev_hist = None

    def is_cut(self, gray):
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        if self.prev_hist is None:
            self.prev_hist = hist
            return False
        dist = cv2.compareHist(self.prev_hist.astype(np.float32),
                               hist.astype(np.float32),
                               cv2.HISTCMP_BHATTACHARYYA)
        self.prev_hist = hist
        return dist > self.thr

# ============================================================
# POST-PROCESSING
# ============================================================
def morph_cleanup(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask

def keep_best_component(mask, h, w):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.zeros_like(mask)

    # Prefer components touching bottom-center region
    y_roi = int(h * 0.75)
    x_roi1, x_roi2 = int(w * 0.35), int(w * 0.65)
    roi_labels = np.unique(labels[y_roi:, x_roi1:x_roi2])
    roi_labels = roi_labels[roi_labels > 0]

    min_area = h * w * MIN_AREA_RATIO
    best_id, best_area = None, 0

    for lid in roi_labels:
        area = stats[lid, cv2.CC_STAT_AREA]
        if area > best_area and area >= min_area:
            best_area, best_id = area, lid

    # Fallback to largest overall if none in ROI
    if best_id is None:
        valid = stats[1:, cv2.CC_STAT_AREA] >= min_area
        if np.any(valid):
            best_id = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        else:
            return np.zeros_like(mask)

    out = np.zeros_like(mask)
    out[labels == best_id] = 255
    return out

# ============================================================
# MAIN PIPELINE
# ============================================================
def process_frame(img_bgr, model, temporal, scene_det, spatial_prior):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # Scene cut detection
    if scene_det.is_cut(gray):
        temporal.reset()
        model.initialized = False

    # Seed region for initialization & online update (bottom-center)
    seed_mask = np.zeros((h, w), dtype=np.uint8)
    y_start = int(h * 0.7)
    x_start, x_end = int(w * 0.3), int(w * 0.7)
    seed_mask[y_start:, x_start:x_end] = 255

    if not model.initialized:
        model._init_from_seed(lab, seed_mask)

    # Cue 1: Color similarity to online model
    color_score = model.score(lab)

    # Cue 2: Texture smoothness
    tex_score = texture_smoothness_score(gray)

    # Cue 3: Spatial prior
    spat_score = spatial_prior

    # Fusion
    fused = (COLOR_WEIGHT * color_score +
             TEXTURE_WEIGHT * tex_score +
             SPATIAL_WEIGHT * spat_score)
    fused = np.clip(fused, 0, 1)

    # Temporal smoothing
    smooth = temporal.update(gray, fused)

    # Online model update (only from high-confidence, low-motion seed area)
    conf_mask = ((smooth > CONFIDENCE_UPDATE_THRESH) & (seed_mask > 0)).astype(np.uint8)
    if np.count_nonzero(conf_mask) > 100:
        model.update(lab, conf_mask)

    # Adaptive thresholding based on seed region statistics
    seed_scores = smooth[seed_mask > 0]
    if len(seed_scores) > 0:
        thr = max(0.35, np.percentile(seed_scores, 40))
    else:
        thr = 0.45

    binary = ((smooth >= thr) * 255).astype(np.uint8)
    cleaned = morph_cleanup(binary)
    final = keep_best_component(cleaned, h, w)

    return final, smooth, seed_mask

# ============================================================
# VISUALIZATION
# ============================================================
def draw_overlay(frame, final_mask, smooth_score, frame_id, fps):
    out = frame.copy()
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(out, "Traversable", (cx-50, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # HUD
    coverage = 100.0 * np.count_nonzero(final_mask) / final_mask.size
    hud = [f"Frame: {frame_id}", f"FPS: {fps:.1f}",
           f"Coverage: {coverage:.1f}%", f"Regions: {len(contours)}"]
    bar_h = 28 * len(hud) + 10
    roi = out[:bar_h, :260].astype(np.float32)
    out[:bar_h, :260] = (roi * 0.4).astype(np.uint8)

    for i, line in enumerate(hud):
        cv2.putText(out, line, (10, 22 + i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return out

# ============================================================
# VIDEO LOOP
# ============================================================
def run_video(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.video}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Downscale for CPU efficiency
    scale = min(1.0, MAX_PROC_WIDTH / W)
    proc_w, proc_h = int(W * scale), int(H * scale)
    disp_size = (proc_w, proc_h)

    print(f"Video: {args.video} | {W}x{H} @ {fps_src:.1f} fps ({total} frames)")
    print(f"Processing at {proc_w}x{proc_h} for CPU efficiency")
    print("Initializing adaptive traversability pipeline...")

    spatial_prior = build_spatial_prior(proc_h, proc_w)
    model = OnlineTraversableModel()
    temporal = TemporalSmoother()
    scene_det = SceneCutDetector()

    frame_id = 0
    t_start = time.time()
    fps_disp = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        final_mask, smooth_score, seed_mask = process_frame(proc, model, temporal, scene_det, spatial_prior)

        fps_disp = 1.0 / max(time.time() - t0, 1e-6)
        vis = draw_overlay(proc, final_mask, smooth_score, frame_id, fps_disp)

        cv2.imshow("Adaptive Traversability (Q=quit)", vis)
        frame_id += 1

        if frame_id % 50 == 0:
            avg_fps = frame_id / (time.time() - t_start)
            print(f"  Frame {frame_id}/{total} | avg {avg_fps:.1f} fps")

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished: {frame_id} frames in {time.time()-t_start:.1f}s")

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CPU-Optimized Adaptive Traversability Detection")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Path to input video")
    args = parser.parse_args()

    print("="*60)
    print("ADAPTIVE TRAVERSABILITY (Indoor/Outdoor, CPU-Optimized)")
    print("Features: Online LAB model, texture cue, spatial prior,")
    print("          motion-gated temporal EMA, no hardcoded colors")
    print("="*60)
    run_video(args)

if __name__ == "__main__":
    main()
