"""
Traversable Path Detection
==========================
Pipeline:
  YOLO11 semantic seg (ADE20K 150 cls)  →  traversable mask
       ↓ every N frames or trigger
  Optical Flow warp                      →  smooth mask between frames
       ↓
  Traditional CV refinement              →  sharpen edges
       ↓
  Connected component + depth limit      →  clean region
       ↓
  Distance-transform centerline          →  path line
       ↓
  Contour + gradient path overlay        →  display
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import sys

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
VIDEO_PATH = "input.mp4"

# ── Model ─────────────────────────────────────────────────────────
# "yolo11n-seg.pt"  → nano,  fastest,  ~2.9M params
# "yolo11s-seg.pt"  → small, balanced, ~10M  params
# Downloads automatically from ultralytics on first run
MODEL_WEIGHT = "yolo11n-seg.pt"

# ── ADE20K class IDs that are traversable ─────────────────────────
# Full 150-class list: https://docs.ultralytics.com/datasets/segment/ade20k/
# We use the most relevant ground-level traversable classes
TRAVERSABLE_IDS: dict[int, str] = {
    3:   "floor",
    6:   "road",
    9:   "grass",
    11:  "sidewalk",
    13:  "earth/ground",
    29:  "field",
    52:  "path",
    53:  "runway",
    55:  "dirt track",
    91:  "sand",
    96:  "gravel",
    121: "straw",
}

# ── YOLO trigger thresholds ────────────────────────────────────────
YOLO_PERIOD          = 8      # Run YOLO every N frames unconditionally
FLOW_CONF_THRESH     = 0.70   # Re-run if optical flow confidence drops below
ROTATION_THRESH_DEG  = 1.5    # Re-run if camera rotates more than this (°)
MASK_SHRINK_THRESH   = 0.45   # Re-run if mask area drops by >45%
YOLO_MIN_CONF        = 0.30   # YOLO detection confidence floor

# ── Temporal blending ─────────────────────────────────────────────
ALPHA_YOLO_NEW       = 0.80   # Weight of fresh YOLO vs warped old mask
ALPHA_TEMPORAL_KEEP  = 0.88   # How much of warped mask to keep each frame

# ── Fusion weights ────────────────────────────────────────────────
W_YOLO               = 0.78   # YOLO semantic mask weight in fusion
W_TRAD               = 0.22   # Traditional CV score weight in fusion

# ── Geometry ──────────────────────────────────────────────────────
# Depth limit: only show path within ~3-4 m
# Empirically the bottom 50% of frame ≈ 3-4 m for typical dash/helmet cam
DEPTH_FRACTION       = 0.50   # 1.0 = full frame height, 0.5 = bottom half
MIN_CONTOUR_AREA     = 900    # Ignore tiny blobs

# ── Display ───────────────────────────────────────────────────────
SHOW_SIDE_PANEL      = True   # Show heatmap + binary mask panel

# ══════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════════
print(f"[INIT] Loading {MODEL_WEIGHT} ...")
model = YOLO(MODEL_WEIGHT)
print(f"[INIT] Model task : {model.task}")
print(f"[INIT] Traversable: {list(TRAVERSABLE_IDS.values())}")


# ══════════════════════════════════════════════════════════════════
#  HELPER: Path colour gradient  (near = blue → far = red)
# ══════════════════════════════════════════════════════════════════
def path_color(progress: float) -> tuple:
    r = int(255 * progress)
    g = int(255 * (1.0 - abs(progress - 0.5) * 2.0))
    b = int(255 * (1.0 - progress))
    return (b, g, r)


# ══════════════════════════════════════════════════════════════════
#  HELPER: Run YOLO semantic segmentation
# ══════════════════════════════════════════════════════════════════
def run_yolo(frame_bgr: np.ndarray,
             out_h: int,
             out_w: int) -> tuple[np.ndarray, list[str], float]:
    """
    Returns
    -------
    prob_mask  : float32 (out_h, out_w)  0-1  traversable probability
    cls_names  : list of detected traversable class names
    mean_conf  : mean confidence of traversable detections
    """
    # ── Run at reduced resolution for CPU speed ────────────────
    infer_w, infer_h = 640, 384
    small = cv2.resize(frame_bgr, (infer_w, infer_h))

    results = model.predict(
        source      = small,
        imgsz       = 640,
        conf        = YOLO_MIN_CONF,
        iou         = 0.45,
        verbose     = False,
        retina_masks= False,   # faster
    )

    prob_mask  = np.zeros((out_h, out_w), dtype=np.float32)
    cls_names  : list[str] = []
    confs      : list[float] = []

    r = results[0]

    # ── Semantic segmentation path ─────────────────────────────
    # r.masks  → instance masks (N, mh, mw)
    # r.boxes  → boxes with cls + conf
    if r.masks is not None and r.boxes is not None:
        masks_np  = r.masks.data.cpu().numpy()          # (N, mh, mw)
        classes   = r.boxes.cls.cpu().numpy().astype(int)
        box_confs = r.boxes.conf.cpu().numpy()

        for idx, cls_id in enumerate(classes):
            if cls_id not in TRAVERSABLE_IDS:
                continue

            conf  = float(box_confs[idx])
            m     = masks_np[idx]                       # (mh, mw) float 0-1

            # Upsample mask to original frame size
            m_up  = cv2.resize(m, (out_w, out_h),
                                interpolation=cv2.INTER_LINEAR)

            # Accumulate weighted probability
            prob_mask = np.maximum(prob_mask, m_up * conf)
            cls_names.append(TRAVERSABLE_IDS[cls_id])
            confs.append(conf)

    mean_conf = float(np.mean(confs)) if confs else 0.0
    return prob_mask, cls_names, mean_conf


# ══════════════════════════════════════════════════════════════════
#  HELPER: Optical flow warp
# ══════════════════════════════════════════════════════════════════
def flow_warp(prev_gray : np.ndarray,
              curr_gray : np.ndarray,
              prev_mask : np.ndarray
              ) -> tuple[np.ndarray, float, float]:
    """
    Estimate affine motion with sparse LK optical flow.
    Returns warped_mask, flow_confidence, rotation_deg.
    """
    # Detect trackable corners in previous frame
    pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners  = 300,
        qualityLevel= 0.01,
        minDistance = 8,
        blockSize   = 7,
    )

    if pts is None or len(pts) < 8:
        return prev_mask, 0.0, 0.0

    pts = pts.reshape(-1, 1, 2).astype(np.float32)

    # Lucas-Kanade pyramid tracking
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None,
        winSize  = (15, 15),
        maxLevel = 3,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    ok          = status.flatten() == 1
    good_prev   = pts[ok].reshape(-1, 2)
    good_curr   = curr_pts[ok].reshape(-1, 2)

    # Confidence from tracking ratio + reprojection error
    track_ratio = float(np.sum(ok)) / max(len(pts), 1)
    if err is not None and np.sum(ok) > 0:
        err_conf = float(np.clip(1.0 - np.mean(err[ok]) / 20.0, 0, 1))
    else:
        err_conf = track_ratio
    flow_conf = track_ratio * 0.6 + err_conf * 0.4

    if np.sum(ok) < 6:
        return prev_mask, float(flow_conf), 0.0

    # Robust affine estimation
    M, _ = cv2.estimateAffinePartial2D(
        good_prev, good_curr,
        method              = cv2.RANSAC,
        ransacReprojThreshold = 3.0,
    )

    if M is None:
        return prev_mask, float(flow_conf) * 0.5, 0.0

    rotation_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))

    h, w   = prev_mask.shape
    warped = cv2.warpAffine(
        prev_mask, M, (w, h),
        flags      = cv2.INTER_LINEAR,
        borderMode = cv2.BORDER_CONSTANT,
        borderValue= 0.0,
    )
    return warped, float(flow_conf), abs(rotation_deg)


# ══════════════════════════════════════════════════════════════════
#  HELPER: Traditional CV score (edge + texture + color)
#          Used ONLY to refine YOLO-confirmed mask edges
# ══════════════════════════════════════════════════════════════════
def trad_score(frame_bgr   : np.ndarray,
               ref_lab_color: np.ndarray) -> np.ndarray:
    """
    Returns float32 (h, w) score in [0, 1].
    ref_lab_color is sampled from YOLO-confirmed traversable pixels.
    """
    lab   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Color similarity to YOLO-confirmed ground patch
    cdist  = np.linalg.norm(lab - ref_lab_color, axis=2)
    cscore = 1.0 - cv2.normalize(cdist, None, 0, 1, cv2.NORM_MINMAX)

    # Low edge density → traversable
    gb     = cv2.GaussianBlur(gray, (5, 5), 0)
    edges  = cv2.Canny(gb, 80, 160).astype(np.float32)
    edges  = cv2.GaussianBlur(edges, (5, 5), 0)
    escore = 1.0 - cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)

    # Low texture → traversable
    blurred  = cv2.GaussianBlur(gray, (7, 7), 0)
    tex      = cv2.absdiff(gray, blurred).astype(np.float32)
    tex      = cv2.blur(tex, (15, 15))
    tscore   = 1.0 - cv2.normalize(tex, None, 0, 1, cv2.NORM_MINMAX)

    score = 0.35 * cscore + 0.40 * escore + 0.25 * tscore
    score = cv2.GaussianBlur(score.astype(np.float32), (11, 11), 0)
    return score


# ══════════════════════════════════════════════════════════════════
#  HELPER: Sample reference LAB color from YOLO-confirmed region
# ══════════════════════════════════════════════════════════════════
def sample_ref_color(frame_bgr: np.ndarray,
                     mask_prob : np.ndarray) -> np.ndarray:
    """
    Pick the bottom 25 % of the YOLO-confirmed mask region.
    These are the most reliable 'ground' pixels.
    """
    h, w = frame_bgr.shape[:2]
    lab  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Restrict to bottom quarter of frame
    bottom = np.zeros((h, w), dtype=np.uint8)
    bottom[int(h * 0.75):, :] = 1

    region = ((mask_prob > 0.40).astype(np.uint8)) * bottom

    if np.sum(region) > 50:
        pixels = lab[region > 0]
        return np.mean(pixels, axis=0)

    # Fallback: plain bottom-centre patch (not road-validated)
    patch = lab[h - 60:h - 10, w // 2 - 40:w // 2 + 40]
    return (np.mean(patch.reshape(-1, 3), axis=0)
            if patch.size > 0
            else np.array([128.0, 128.0, 128.0]))


# ══════════════════════════════════════════════════════════════════
#  HELPER: Extract depth-limited centerline path
# ══════════════════════════════════════════════════════════════════
def extract_path(traversable : np.ndarray,
                 seed_y      : int,
                 limit_y     : int,
                 w           : int) -> list[tuple[int, int]]:
    """
    Trace the medial axis of `traversable` from seed_y upward to limit_y.
    Returns smoothed list of (x, y) points.
    """
    dist = cv2.distanceTransform(traversable, cv2.DIST_L2, 5)
    raw  : list[tuple[int, int]] = []

    for y in range(seed_y, limit_y, -4):
        row = dist[y, :]
        if row.max() == 0:
            continue

        # Centre bias: penalise horizontal deviation from image centre
        bias      = np.abs(np.arange(w) - w // 2) * 0.04
        best_x    = int(np.argmax(row - bias))

        if traversable[y, best_x] > 0:
            raw.append((best_x, y))

    if len(raw) < 4:
        return raw

    # Sliding-window smooth
    pts, win = [], 7
    for i in range(len(raw)):
        s     = max(0, i - win)
        e     = min(len(raw), i + win + 1)
        avg_x = int(np.mean([p[0] for p in raw[s:e]]))
        pts.append((avg_x, raw[i][1]))

    return pts


# ══════════════════════════════════════════════════════════════════
#  HELPER: Draw everything on frame
# ══════════════════════════════════════════════════════════════════
def draw_overlay(frame        : np.ndarray,
                 traversable  : np.ndarray,
                 path_pts     : list,
                 cls_names    : list[str],
                 yolo_ran     : bool,
                 flow_conf    : float,
                 fps          : float,
                 limit_y      : int) -> np.ndarray:

    h, w  = frame.shape[:2]
    out   = frame.copy()

    # ── Translucent green fill ─────────────────────────────────
    overlay      = out.copy()
    fill         = np.zeros_like(out)
    fill[traversable > 0] = (30, 190, 30)
    cv2.addWeighted(fill, 0.28, overlay, 0.72, 0, out)

    # ── Depth-limit horizon line ───────────────────────────────
    cv2.line(out, (0, limit_y), (w, limit_y), (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "~3-4 m limit",
                (6, limit_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 200, 255), 1, cv2.LINE_AA)

    # ── Contour ────────────────────────────────────────────────
    cnts, _ = cv2.findContours(
        traversable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) > MIN_CONTOUR_AREA:
            cv2.drawContours(out, [largest], -1, (0, 255, 80), 2, cv2.LINE_AA)

    # ── Gradient path line ─────────────────────────────────────
    if len(path_pts) > 1:
        n = len(path_pts)
        for i in range(n - 1):
            prog  = i / n
            col   = path_color(prog)
            thick = max(2, int(8 * (1.0 - prog * 0.65)))
            cv2.line(out, path_pts[i], path_pts[i + 1],
                     col, thick, cv2.LINE_AA)

        # Arrow at farthest visible point
        if n >= 3:
            cv2.arrowedLine(out, path_pts[-3], path_pts[-1],
                            (0, 220, 255), 2,
                            tipLength=0.35, line_type=cv2.LINE_AA)

    # ── HUD ───────────────────────────────────────────────────
    def put(txt, y, col=(255, 255, 255), scale=0.58, bold=1):
        cv2.putText(out, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col, bold, cv2.LINE_AA)

    put(f"FPS  {fps:5.1f}",           26, (0, 255, 255), 0.65, 2)
    put(f"Flow {flow_conf:.2f}",      50, (200, 200, 200))
    put(f"YOLO {'RAN' if yolo_ran else 'skip'}",
                                      72,
                                      (0, 80, 255) if yolo_ran else (120, 120, 120),
                                      bold=2)
    if cls_names:
        label = ", ".join(sorted(set(cls_names)))
        put(f"Cls  {label}",          94, (80, 255, 80))

    put(f"Path {len(path_pts)} pts",
        h - 12, (255, 255, 255), 0.50)

    return out


# ══════════════════════════════════════════════════════════════════
#  OPEN VIDEO
# ══════════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    sys.exit(f"[ERROR] Cannot open: {VIDEO_PATH}")

# ══════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════
frame_idx        = 0
prev_gray        = None

# Float probability masks  (h, w)  range [0, 1]
blended_mask     : np.ndarray | None = None   # working temporal mask
prev_mask_area   = 0                          # for shrink detection

# Reference surface colour (LAB 3-vector) — seeded by YOLO
ref_lab          = np.array([128.0, 128.0, 128.0])
yolo_confirmed   = False           # True once YOLO validates a traversable region

prev_trad_score  : np.ndarray | None = None   # temporal smoothing
last_cls_names   : list[str]  = []
last_flow_conf   = 1.0
last_yolo_ran    = False

fps_q   = deque(maxlen=30)
t_prev  = time.perf_counter()

morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

print("[MAIN] Running — press Q to quit")

# ══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w      = frame.shape[:2]
    seed_y    = h - 15
    seed_x    = w // 2

    # Depth-limit horizon (pixels above this are "> 4 m away")
    limit_y   = int(h * (1.0 - DEPTH_FRACTION))

    # ROI: only below the depth-limit line matters
    roi       = np.zeros((h, w), dtype=np.uint8)
    roi[limit_y:, :] = 255

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── STEP 1: Optical flow warp ──────────────────────────────
    flow_conf    = 1.0
    rotation_deg = 0.0
    warped_mask  = blended_mask

    if prev_gray is not None and blended_mask is not None:
        warped_mask, flow_conf, rotation_deg = flow_warp(
            prev_gray, curr_gray, blended_mask
        )

    last_flow_conf = flow_conf

    # ── STEP 2: Decide whether to run YOLO ────────────────────
    curr_area  = int(np.sum(blended_mask > 0.4)) if blended_mask is not None else 0
    shrink_ok  = (curr_area / max(prev_mask_area, 1)) < (1.0 - MASK_SHRINK_THRESH) \
                 if prev_mask_area > 0 else False

    need_yolo = (
        blended_mask is None           # first frame
        or frame_idx % YOLO_PERIOD == 0
        or flow_conf    < FLOW_CONF_THRESH
        or rotation_deg > ROTATION_THRESH_DEG
        or shrink_ok
    )

    yolo_ran  = False
    found_cls : list[str] = []

    if need_yolo:
        yolo_prob, found_cls, yolo_conf = run_yolo(frame, h, w)

        # Apply ROI (discard predictions beyond depth limit)
        yolo_prob *= (roi.astype(np.float32) / 255.0)

        if found_cls:
            # YOLO confirmed a traversable surface
            ref_lab        = sample_ref_color(frame, yolo_prob)
            yolo_confirmed = True
            last_cls_names = found_cls

            # Blend fresh YOLO result with warped historical mask
            if warped_mask is not None:
                blended_mask = (ALPHA_YOLO_NEW       * yolo_prob +
                                (1 - ALPHA_YOLO_NEW) * warped_mask)
            else:
                blended_mask = yolo_prob.copy()

            yolo_ran = True
            print(f"  [YOLO] f={frame_idx:04d} "
                  f"cls={found_cls} conf={yolo_conf:.2f} "
                  f"flow={flow_conf:.2f} rot={rotation_deg:.1f}°")
        else:
            # YOLO ran but found nothing → decay warped mask
            blended_mask = (warped_mask * 0.65
                            if warped_mask is not None
                            else np.zeros((h, w), dtype=np.float32))
            print(f"  [YOLO] f={frame_idx:04d} "
                  f"NOTHING traversable  flow={flow_conf:.2f}")
    else:
        # Non-YOLO frame: keep warped mask with slight decay
        if warped_mask is not None:
            blended_mask = (ALPHA_TEMPORAL_KEEP       * warped_mask +
                            (1 - ALPHA_TEMPORAL_KEEP) *
                            (blended_mask if blended_mask is not None
                             else warped_mask))

    # Safety
    if blended_mask is None:
        blended_mask = np.zeros((h, w), dtype=np.float32)

    blended_mask = np.clip(
        blended_mask * (roi.astype(np.float32) / 255.0),
        0.0, 1.0
    )

    # ── STEP 3: Fuse YOLO mask with traditional CV score ──────
    # Traditional score is calibrated to YOLO-confirmed surface colour
    # so it REFINES edges, not bootstraps from scratch
    t_score = trad_score(frame, ref_lab)

    if prev_trad_score is not None:
        t_score = 0.65 * t_score + 0.35 * prev_trad_score
    prev_trad_score = t_score.copy()

    if yolo_confirmed:
        fused = W_YOLO * blended_mask + W_TRAD * t_score
    else:
        # Haven't confirmed anything yet → use only traditional
        # but DON'T draw path (no semantic validation)
        fused = t_score.copy()

    fused = cv2.GaussianBlur(fused.astype(np.float32), (7, 7), 0)

    # ── STEP 4: Threshold → binary mask ───────────────────────
    roi_vals  = fused[roi > 0]
    threshold = float(np.percentile(roi_vals, 58)) if len(roi_vals) > 0 else 0.45

    binary = (fused > threshold).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, roi)

    # Morphological cleanup
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  morph_k, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_k, iterations=3)

    # ── STEP 5: Connected component — seed-based selection ────
    traversable = np.zeros((h, w), dtype=np.uint8)

    if yolo_confirmed:
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        seed_lbl = int(labels[seed_y, seed_x])

        if seed_lbl > 0:
            # Camera is directly above a traversable component
            traversable[labels == seed_lbl] = 255
        else:
            # Seed pixel not in any component →
            # pick largest component touching the bottom strip
            strip_y = int(h * 0.82)
            bottom_set = set(np.unique(labels[strip_y:, :]))

            best_lbl, best_area = -1, 0
            for lbl in range(1, n_lbl):
                if lbl in bottom_set:
                    area = int(stats[lbl, cv2.CC_STAT_AREA])
                    if area > best_area:
                        best_area = area
                        best_lbl  = lbl

            if best_lbl > 0:
                traversable[labels == best_lbl] = 255

    prev_mask_area = int(np.sum(traversable > 0))

    # ── STEP 6: Depth-limited centerline path ─────────────────
    path_pts: list[tuple[int, int]] = []
    if yolo_confirmed and prev_mask_area > MIN_CONTOUR_AREA:
        path_pts = extract_path(traversable, seed_y, limit_y, w)

    # ── STEP 7: FPS ───────────────────────────────────────────
    t_now    = time.perf_counter()
    fps_q.append(1.0 / max(t_now - t_prev, 1e-6))
    t_prev   = t_now
    avg_fps  = float(np.mean(fps_q))

    # ── STEP 8: Render ────────────────────────────────────────
    vis = draw_overlay(
        frame, traversable, path_pts,
        last_cls_names, yolo_ran,
        last_flow_conf, avg_fps, limit_y,
    )

    if SHOW_SIDE_PANEL:
        # Left: annotated result
        # Right top: YOLO probability heatmap
        heat = cv2.applyColorMap(
            (blended_mask * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO
        )
        heat = cv2.addWeighted(frame, 0.35, heat, 0.65, 0)
        cv2.putText(heat, "YOLO+flow prob",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 2, cv2.LINE_AA)

        # Right bottom: binary traversable mask
        trav_bgr = cv2.cvtColor(traversable, cv2.COLOR_GRAY2BGR)

        # Flow confidence bar
        bar_len = int(last_flow_conf * (w - 20))
        bar_col = ((0, 220, 0)   if last_flow_conf > 0.7 else
                   (0, 165, 255) if last_flow_conf > 0.4 else
                   (0, 0, 220))
        cv2.rectangle(trav_bgr,
                      (10, h - 18), (10 + bar_len, h - 6),
                      bar_col, -1)
        cv2.putText(trav_bgr,
                    f"Traversable binary | flow {last_flow_conf:.2f}",
                    (10, h - 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (200, 200, 200), 1, cv2.LINE_AA)

        target = (640, 480)
        top    = np.hstack([cv2.resize(vis,      target),
                            cv2.resize(heat,     target)])
        bot    = np.hstack([cv2.resize(trav_bgr, target),
                            np.zeros((target[1], target[0], 3), np.uint8)])
        display = np.vstack([top, bot])
    else:
        display = cv2.resize(vis, (1280, 720))

    cv2.imshow("Traversable Path  [YOLO-sem + Flow]", display)

    prev_gray  = curr_gray.copy()
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quit.")
        break

cap.release()
cv2.destroyAllWindows()
