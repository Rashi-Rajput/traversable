# Save this as traversability_accurate.py
import os
import argparse
import time
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from collections import deque
from skimage.segmentation import slic
from skimage.color import rgb2lab

# ----------------------------- CONFIG -----------------------------
DEFAULT_VIDEO = "vid.mp4"

GMM_COMPONENTS = 4
LIKELIHOOD_PERCENTILE = 20
SKY_CUTOFF_RATIO = 0.52
MORPH_KERNEL_SIZE = 11
MIN_CONTOUR_AREA = 2500

TEMPORAL_WINDOW = 7
TEMPORAL_ALPHA = 0.6
TEMPORAL_THRESHOLD = 0.42

TRAVERSABLE_BGR = np.array([
    [80, 130, 70], [90, 145, 80], [105, 160, 90],
    [70, 120, 65], [120, 170, 100], [120, 90, 60],
    [135, 100, 70], [150, 115, 80], [100, 75, 55],
    [170, 130, 90], [150, 145, 130], [165, 160, 145],
    [180, 175, 160], [130, 125, 115], [190, 185, 170],
    [70, 75, 80], [85, 90, 95], [100, 105, 110],
    [55, 60, 65], [120, 125, 130], [135, 135, 130],
    [150, 150, 145], [165, 165, 160], [180, 180, 175],
    [110, 110, 105], [110, 120, 100], [125, 135, 115],
    [140, 145, 125], [95, 110, 90], [155, 150, 135],
], dtype=np.float32)

# ============================================================
# TEMPORAL CONSISTENCY (Improved)
# ============================================================
class TemporalConsistency:
    def __init__(self, window=TEMPORAL_WINDOW, alpha=TEMPORAL_ALPHA):
        self.window = window
        self.alpha = alpha
        self.temporal_score = None
        self.prev_gray = None
        self.score_buffer = deque(maxlen=window)
        self.flow_params = dict(
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

    def _warp(self, score_map, flow):
        h, w = flow.shape[:2]
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        map_x = gx + flow[..., 0]
        map_y = gy + flow[..., 1]
        return cv2.remap(score_map, map_x.astype(np.float32),
                         map_y.astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def update(self, curr_gray, curr_score):
        if self.temporal_score is None:
            self.temporal_score = curr_score.copy()
            self.prev_gray = curr_gray.copy()
            self.score_buffer.append(curr_score.copy())
            return curr_score.copy()

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, curr_gray, None, **self.flow_params)

        warped = self._warp(self.temporal_score, flow)
        smooth = np.clip(
            self.alpha * curr_score + (1 - self.alpha) * warped, 0, 1)

        self.temporal_score = smooth.copy()
        self.prev_gray = curr_gray.copy()
        self.score_buffer.append(curr_score.copy())

        # Median smoothing over recent frames
        if len(self.score_buffer) >= 3:
            median_score = np.median(list(self.score_buffer), axis=0)
            smooth = 0.7 * smooth + 0.3 * median_score

        return smooth

    def reset(self):
        self.temporal_score = None
        self.prev_gray = None
        self.score_buffer.clear()

# ============================================================
# SCENE CUT DETECTOR
# ============================================================
class SceneCutDetector:
    def __init__(self, threshold=0.42):
        self.threshold = threshold
        self.prev_hist = None

    def is_cut(self, gray):
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        if self.prev_hist is None:
            self.prev_hist = hist
            return False
        dist = cv2.compareHist(self.prev_hist.astype(np.float32),
                               hist.astype(np.float32),
                               cv2.HISTCMP_BHATTACHARYYA)
        self.prev_hist = hist
        return dist > self.threshold

# ============================================================
# HELPERS
# ============================================================
def get_improved_ground_mask(shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    y_start = int(h * SKY_CUTOFF_RATIO)
    mask[y_start:, :] = 255
    return mask, y_start

def normalize01(x, mask=None):
    x = x.astype(np.float32)
    if mask is not None and np.any(mask):
        vals = x[mask]
        mn, mx = vals.min(), vals.max()
    else:
        mn, mx = x.min(), x.max()
    if mx - mn < 1e-6:
        return np.zeros_like(x)
    out = (x - mn) / (mx - mn)
    out = np.clip(out, 0, 1)
    if mask is not None:
        out[~mask] = 0
    return out

# ============================================================
# IMPROVED GMM (Superpixel + richer features)
# ============================================================
def build_gmm(traversable_bgr):
    db = traversable_bgr.astype(np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(db, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(db, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    bgrn = traversable_bgr / 255.0
    feat = np.hstack([hsv, lab, bgrn]).astype(np.float32)

    gmm = GaussianMixture(n_components=GMM_COMPONENTS,
                          covariance_type="full", reg_covar=1e-3,
                          max_iter=600, random_state=42)
    gmm.fit(feat)
    scores = gmm.score_samples(feat)
    thr = np.percentile(scores, LIKELIHOOD_PERCENTILE)
    return gmm, thr

def gmm_superpixel_mask(img_bgr, gmm, thr, ground_mask):
    h, w = img_bgr.shape[:2]
    lab = rgb2lab(img_bgr[:, :, ::-1])
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bgrn = img_bgr.astype(np.float32) / 255.0
    feat_img = np.dstack([hsv, lab, bgrn]).astype(np.float32)

    segments = slic(img_bgr, n_segments=400, compactness=12, sigma=1)
    mask = np.zeros((h, w), dtype=np.uint8)

    for seg_id in np.unique(segments):
        seg_mask = segments == seg_id
        if not np.any(seg_mask & (ground_mask > 0)):
            continue
        mean_feat = feat_img[seg_mask].mean(axis=0, keepdims=True)
        if gmm.score_samples(mean_feat)[0] > thr:
            mask[seg_mask] = 255
    return cv2.bitwise_and(mask, ground_mask)

# ============================================================
# COLOR TEMPLATE + ADAPTIVE (Improved)
# ============================================================
def color_template_mask(img_bgr, traversable_bgr, ground_mask):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    db_lab = cv2.cvtColor(traversable_bgr.astype(np.uint8).reshape(-1,1,3),
                          cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float32)

    min_dist = np.full(img_bgr.shape[:2], np.inf, dtype=np.float32)
    for c in db_lab:
        min_dist = np.minimum(min_dist, np.linalg.norm(lab - c, axis=2))

    valid = ground_mask > 0
    heat = 1.0 - normalize01(min_dist, valid)
    heat[~valid] = 0
    thr = max(0.38, np.percentile(heat[valid], 52) if np.any(valid) else 0.5)
    mask = ((heat >= thr) * 255).astype(np.uint8)
    return heat, cv2.bitwise_and(mask, ground_mask)

# ============================================================
# FUSION (Improved)
# ============================================================
def fuse_masks(hsv_m, template_m, gmm_m, adaptive_m, template_heat, ground_mask):
    hsv_b = (hsv_m > 0).astype(np.float32)
    tmpl_b = (template_m > 0).astype(np.float32)
    gmm_b = (gmm_m > 0).astype(np.float32)
    adap_b = (adaptive_m > 0).astype(np.float32)

    score = (0.18 * hsv_b +
             0.22 * template_heat.astype(np.float32) +
             0.28 * gmm_b +
             0.32 * adap_b)

    votes = hsv_b + tmpl_b + gmm_b + adap_b
    fused = np.zeros_like(hsv_m, dtype=np.uint8)
    fused[(score >= 0.48) | ((votes >= 2) & (score > 0.35))] = 255
    return cv2.bitwise_and(fused, ground_mask), score

# ============================================================
# POST-PROCESSING (Improved)
# ============================================================
def morph_cleanup(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    return mask

def keep_best_component(mask, seed):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.zeros_like(mask)

    h, w = mask.shape
    sx, sy = int(np.clip(seed[0], 0, w-1)), int(np.clip(seed[1], 0, h-1))
    sl = labels[sy, sx]

    out = np.zeros_like(mask)
    if sl > 0 and stats[sl, cv2.CC_STAT_AREA] >= MIN_CONTOUR_AREA:
        out[labels == sl] = 255
        return out

    # Fallback to largest valid component in lower center
    yb1 = max(h-120, 0)
    xb1, xb2 = max(w//2 - w//4, 0), min(w//2 + w//4, w)
    cands = np.unique(labels[yb1:h, xb1:xb2])
    cands = cands[cands > 0]

    best_id, best_area = None, 0
    for lid in cands:
        a = stats[lid, cv2.CC_STAT_AREA]
        if a > best_area and a >= MIN_CONTOUR_AREA:
            best_area, best_id = a, lid

    if best_id is not None:
        out[labels == best_id] = 255
    return out

# ============================================================
# MAIN PIPELINE
# ============================================================
def process_frame(img_bgr, gmm, gmm_thr):
    ground_mask, _ = get_improved_ground_mask(img_bgr.shape)

    hsv_m = cv2.bitwise_and(
        cv2.inRange(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV),
                    np.array([20, 25, 30]), np.array([100, 255, 255])),
        ground_mask)

    gmm_m = gmm_superpixel_mask(img_bgr, gmm, gmm_thr, ground_mask)
    t_heat, t_mask = color_template_mask(img_bgr, TRAVERSABLE_BGR, ground_mask)

    # Simple adaptive mask (improved)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edge_d = cv2.blur(edges.astype(np.float32), (21, 21))
    edge_d = normalize01(edge_d, ground_mask > 0)
    a_score = 1.0 - edge_d
    a_mask = ((a_score > 0.48) * 255).astype(np.uint8)
    a_mask = cv2.bitwise_and(a_mask, ground_mask)

    fused, fused_score = fuse_masks(hsv_m, t_mask, gmm_m, a_mask, t_heat, ground_mask)
    return fused_score, ground_mask

def apply_temporal(smooth_score, ground_mask):
    binary = ((smooth_score >= TEMPORAL_THRESHOLD) * 255).astype(np.uint8)
    binary = cv2.bitwise_and(binary, ground_mask)
    cleaned = morph_cleanup(binary)
    return keep_best_component(cleaned, (img_bgr.shape[1]//2, img_bgr.shape[0]-30))

# ============================================================
# VISUALIZATION
# ============================================================
def draw_contours_only(frame, final_mask, frame_id, fps):
    out = frame.copy()
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cv2.drawContours(out, contours, -1, (255, 255, 255), 5)
        cv2.drawContours(out, contours, -1, (0, 255, 0), 2)

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(out, "Traversable", (cx-60, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # HUD
    coverage = 100.0 * np.count_nonzero(final_mask) / final_mask.size
    hud = [f"Frame: {frame_id}", f"FPS: {fps:.1f}",
           f"Coverage: {coverage:.1f}%", f"Regions: {len(contours)}"]
    bar_h = 30 * len(hud) + 8
    roi = out[:bar_h, :280].astype(np.float32)
    out[:bar_h, :280] = (roi * 0.35).astype(np.uint8)

    for i, line in enumerate(hud):
        cv2.putText(out, line, (10, 26 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
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

    print(f"Video: {args.video} | {W}x{H} @ {fps_src:.1f} fps ({total} frames)")
    print("Building improved GMM ...", end=" ", flush=True)
    gmm, gmm_thr = build_gmm(TRAVERSABLE_BGR)
    print(f"done (thr={gmm_thr:.4f})")

    temporal = TemporalConsistency(window=args.temporal_window, alpha=args.temporal_alpha)
    scene_det = SceneCutDetector()

    max_w = 1280
    scale = min(1.0, max_w / W)
    disp_size = (int(W * scale), int(H * scale))

    frame_id = 0
    t_start = time.time()
    fps_disp = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if scene_det.is_cut(gray):
            temporal.reset()
            print(f"  [f{frame_id}] Scene cut → temporal reset")

        fused_score, ground_mask = process_frame(frame, gmm, gmm_thr)
        smooth = temporal.update(gray, fused_score)
        final_mask = apply_temporal(smooth, ground_mask)

        fps_disp = 1.0 / max(time.time() - t0, 1e-6)
        vis = draw_contours_only(frame, final_mask, frame_id, fps_disp)

        disp = cv2.resize(vis, disp_size) if scale < 1.0 else vis
        cv2.imshow("Accurate Traversability (Q=quit)", disp)

        frame_id += 1
        if frame_id % 40 == 0:
            print(f"  Frame {frame_id}/{total} | avg {frame_id/(time.time()-t_start):.1f} fps")

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished: {frame_id} frames in {time.time()-t_start:.1f}s")

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--temporal-window", type=int, default=TEMPORAL_WINDOW)
    parser.add_argument("--temporal-alpha", type=float, default=TEMPORAL_ALPHA)
    args = parser.parse_args()

    print("="*60)
    print("IMPROVED Traversability Detection (Superpixels + Better Fusion)")
    print("="*60)
    run_video(args)

if __name__ == "__main__":
    main()