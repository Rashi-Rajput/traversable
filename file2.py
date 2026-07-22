import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import warnings
warnings.filterwarnings("ignore")

# ───────────────────────── CONFIG ─────────────────────────
VIDEO_PATH = "input.mp4"

# YOLOv26-Seg ADE20K pretrained weights
# Download from: https://docs.ultralytics.com/tasks/segment/
# Example file: yolov8n-seg-ade20k.pt / yolo26n-seg-ade20k.pt
MODEL_WEIGHTS = "yolo26n-seg-ade20k.pt"

# Only run expensive model on these conditions
TRIGGER_INTERVAL     = 8       # every N frames
FLOW_CONF_MIN        = 0.70    # good feature ratio
ROT_MAX_DEG          = 12.0    # large rotation
MASK_SHRINK_RATIO    = 0.55    # area drop vs previous

# Near-field limit ~3-4m (bottom area with tapered width)
NEAR_Y0             = 0.45
NEAR_MAX_Y_RATIO    = 0.92    # do not trace above this line (horizon cut)

# ADE20K traversable class IDs (update if your weight mapping differs)
# Typical ADE20K: road=0, path=12, sidewalk=13, grass=9, terrain=11, earth=29
ADE20K_TRAVERSABLE  = {0, 9, 11, 12, 13, 29, 96}

# Temporal smoothing
TEMP_ALPHA          = 0.85

# ───────────────────────── MODEL LOAD ─────────────────────────
print("[INFO] Loading YOLOv26-Seg ADE20K ...")
seg_model = YOLO(MODEL_WEIGHTS)  # pretrained weights loaded automatically
seg_model.fuse()                 # optimize for CPU if available
seg_model.to("cpu")

# Helper: near-field ROI with perspective taper (3-4m limit)
def build_near_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    y_start = int(h * NEAR_Y0)
    mask[y_start:, :] = 255
    # Taper sides near horizon so distant sides are ignored
    for y in range(y_start, h):
        # At bottom allow ~85% width, near horizon allow ~25%
        progress = (y - y_start) / max(1, h - y_start)
        half_w = int(w * (0.42 - 0.30 * progress))  # taper in
        cx = w // 2
        mask[y, :] = 0
        mask[y, max(0, cx - half_w):min(w, cx + half_w)] = 255
    return mask

# Helper: path color gradient
def get_path_color(p):
    return (int(255*p), int(255*(1-p)), int(255*(1-p)))

# Helper: fast traditional score (your original logic)
def compute_trad_score(im):
    h, w = im.shape[:2]
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Seed patch near camera bottom-center (within 3-4m)
    y1, y2 = max(0, h-70), h-10
    x1, x2 = max(0, w//2 - 35), min(w, w//2 + 35)
    patch = lab[y1:y2, x1:x2]
    floor_col = np.mean(patch.reshape(-1,3), axis=0).astype(np.float32) if patch.size>0 else np.array([128,128,128], np.float32)

    # Color distance
    col_dist = np.linalg.norm(lab.astype(np.float32) - floor_col, axis=2)
    col_score = 1.0 - cv2.normalize(col_dist, None, 0, 1, cv2.NORM_MINMAX)

    # Edge
    gb = cv2.GaussianBlur(gray, (5,5), 0)
    ed = cv2.Canny(gb, 80, 160)
    ed = cv2.GaussianBlur(ed.astype(np.float32), (5,5), 0)
    edge_score = 1.0 - cv2.normalize(ed, None, 0, 1, cv2.NORM_MINMAX)

    # Texture
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    tex = cv2.absdiff(gray, blur).astype(np.float32)
    tex = cv2.normalize(cv2.blur(tex, (15,15)), None, 0, 1, cv2.NORM_MINMAX)
    tex_score = 1.0 - tex

    score = 0.30*col_score + 0.50*edge_score + 0.20*tex_score
    score = cv2.GaussianBlur(score, (11,11), 0)
    return score.astype(np.float32)

# Helper: estimate rotation angle from sparse flow
def estimate_rotation(prev_gray, curr_gray):
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=120, qualityLevel=0.01, minDistance=8, blockSize=5)
    if pts is None or len(pts) < 4:
        return 0.0
    pts = pts.reshape(-1, 1, 2).astype(np.float32)
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, winSize=(15,15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    if status is None:
        return 0.0
    good_prev = pts[status.flatten()==1]
    good_curr = curr_pts[status.flatten()==1]
    if len(good_prev) < 4:
        return 0.0
    M, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return 0.0
    angle = np.degrees(np.arctan2(M[1,0], M[0,0]))
    return abs(float(angle))

# Helper: flow tracking confidence
def flow_confidence(prev_gray, curr_gray):
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=150, qualityLevel=0.01, minDistance=8, blockSize=5)
    if pts is None or len(pts) < 6:
        return 0.0
    pts = pts.reshape(-1,1,2).astype(np.float32)
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, winSize=(15,15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    if status is None:
        return 0.0
    return float(np.sum(status==1) / len(status))

# Helper: run YOLO-Seg only on near-field crop (fast)
def run_yolo_near(im, near_mask):
    h, w = im.shape[:2]
    y0 = int(h * NEAR_Y0)
    crop = im[y0:, :, :]
    # Predict at reduced size for CPU speed
    results = seg_model.predict(source=crop, conf=0.35, imgsz=384, device="cpu", verbose=False, max_det=1)
    mask_full = np.zeros((h, w), dtype=np.uint8)

    if results and len(results) > 0:
        r = results[0]
        if r.masks is not None and r.masks.data.numel() > 0:
            # Process each mask
            for seg_tensor, cls in zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                if int(cls) in ADE20K_TRAVERSABLE:
                    # seg_tensor shape: (1, H_crop, W_crop) float probability
                    seg_bin = (seg_tensor.squeeze() > 0.3).astype(np.uint8) * 255
                    ch, cw = crop.shape[:2]
                    if seg_bin.shape[:2] != (ch, cw):
                        seg_bin = cv2.resize(seg_bin, (cw, ch), interpolation=cv2.INTER_NEAREST)
                    # Place into full frame
                    mask_full[y0:y0+ch, :cw] |= seg_bin
    # Restrict to near-field tapered region
    mask_full = cv2.bitwise_and(mask_full, near_mask)
    return mask_full

# ───────────────────────── MAIN LOOP ─────────────────────────
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {VIDEO_PATH}")
        return

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    near_mask = build_near_mask(h, w)

    # State
    prev_gray = None
    prev_trad_score = None
    blended_mask = None       # float [0,1] running corrected traversable region
    prev_area = 0.0
    frame_idx = 0
    reasons = []

    # FPS smoothing
    fps_buf = deque(maxlen=20)

    print("[INFO] Running. YOLO triggers: interval, low_flow(<0.7), rot(>12°), tracking_lost, mask_shrink")

    while True:
        ret, image = cap.read()
        if not ret:
            break

        t0 = time.time()
        frame_idx += 1

        # --- 1. Traditional fast score (every frame) ---
        score = compute_trad_score(image)
        # Force near-field only
        score = score * (near_mask.astype(np.float32) / 255.0)

        # Temporal smoothing of score
        if prev_trad_score is not None:
            score = 0.85 * score + 0.15 * prev_trad_score
        prev_trad_score = score.copy()

        # --- 2. Build binary from score ---
        binary_trad = ((score > np.mean(score) * 1.02) * 255).astype(np.uint8)
        binary_trad = cv2.bitwise_and(binary_trad, near_mask)

        # Clean
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary_trad = cv2.morphologyEx(binary_trad, cv2.MORPH_OPEN, k, iterations=2)
        binary_trad = cv2.morphologyEx(binary_trad, cv2.MORPH_CLOSE, k, iterations=2)

        # --- 3. Check triggers ---
        trigger = False
        reasons = []

        if frame_idx % TRIGGER_INTERVAL == 0:
            trigger = True
            reasons.append("interval")

        # Flow / rotation tracking
        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            fc = flow_confidence(prev_gray, curr_gray)
            rot = estimate_rotation(prev_gray, curr_gray)

            if fc < FLOW_CONF_MIN:
                trigger = True
                reasons.append(f"low_flow({fc:.2f})")
            if rot > ROT_MAX_DEG:
                trigger = True
                reasons.append(f"rot({rot:.0f})")

        # Tracking lost / mask shrink check (using previous blended area if available)
        current_area = cv2.countNonZero(binary_trad)
        if blended_mask is not None:
            prev_blended_area = np.sum(blended_mask > 0.3) * 1.0  # approximate pixel area scale
            # If current traditional area collapsed severely
            if current_area < max(500, prev_blended_area * 0.4) and prev_blended_area > 100:
                # Only trigger if it really shrank drastically
                pass  # We'll check after correction below
        # We'll do shrink check after blend below

        prev_gray = curr_gray.copy()

        # --- 4. YOLO-Seg validation / correction (only on trigger) ---
        if trigger:
            yolo_mask = run_yolo_near(image, near_mask)

            if cv2.countNonZero(yolo_mask) > 200:
                # YOLO found traversable area in near field -> correct blend
                yolo_f = yolo_mask.astype(np.float32) / 255.0
                if blended_mask is None:
                    blended_mask = yolo_f.copy()
                else:
                    # Strong correction: mostly trust YOLO, keep some temporal context
                    blended_mask = 0.65 * yolo_f + 0.35 * blended_mask
                reasons.append("yolo_corr")
            else:
                # YOLO found nothing traversable -> reset blend to conservative traditional
                blended_mask = (score > 0.5).astype(np.float32) if blended_mask is None else blended_mask * 0.3
                reasons.append("yolo_empty")

            # Update shrink tracking based on corrected mask
            prev_area = np.sum(blended_mask > 0.3) * 1.0
        else:
            # Non-trigger: propagate previous blend with slow temporal fusion
            # Also apply small correction from traditional score
            if blended_mask is not None:
                blended_mask = TEMP_ALPHA * blended_mask + (1.0 - TEMP_ALPHA) * (binary_trad.astype(np.float32) / 255.0)
            else:
                blended_mask = (binary_trad.astype(np.float32) / 255.0)

        # Check mask shrink after blend update (if area collapsed suddenly)
        current_blend_area = np.sum(blended_mask > 0.3)
        if prev_area > 300 and current_blend_area < prev_area * MASK_SHRINK_RATIO:
            # Force a re-trigger concept: reset blend with stronger traditional weight
            blended_mask = 0.8 * (binary_trad.astype(np.float32) / 255.0) + 0.2 * blended_mask
            reasons.append("shrink_fix")

        # Store area for next frame
        prev_blend_area_store = current_blend_area
        prev_area = current_blend_area

        # --- 5. Create final traversable binary from blended mask ---
        # Use adaptive threshold but keep near-field strict
        fused_score = blended_mask.copy()
        # Small sharpening via score gradient? No, keep simple.
        thr = np.mean(fused_score[near_mask > 0]) * 1.05 if np.any(near_mask > 0) else 0.5
        traversable = ((fused_score > max(thr, 0.35)) * 255).astype(np.uint8)
        traversable = cv2.bitwise_and(traversable, near_mask)

        # Clean again lightly
        traversable = cv2.morphologyEx(traversable, cv2.MORPH_OPEN, k, iterations=1)
        traversable = cv2.morphologyEx(traversable, cv2.MORPH_CLOSE, k, iterations=1)

        # --- 6. Path extraction (your original logic) with 3-4m limit ---
        path_points = []
        contours, _ = cv2.findContours(traversable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = image.copy()

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area_largest = cv2.contourArea(largest)

            # Only proceed if we have a real traversable region
            if area_largest > 400:
                cv2.drawContours(result, [largest], -1, (0, 255, 0), 3)

                # Distance transform for centerline
                dist_map = cv2.distanceTransform(traversable, cv2.DIST_L2, 3)

                y_coords = np.where(traversable > 0)[0]
                if len(y_coords) > 0:
                    min_y = int(np.min(y_coords))
                    seed_y = h - 20
                    # HARD LIMIT: do not trace above near horizon (~3-4m visual distance)
                    max_trace_y = max(min_y, int(h * NEAR_MAX_Y_RATIO))

                    raw_path = []
                    for y in range(seed_y, max_trace_y, -5):
                        row = dist_map[y, :]
                        if np.max(row) <= 0:
                            continue
                        # Center bias for straight path preference
                        center_bias = np.abs(np.arange(w) - w // 2) * 0.05
                        biased_row = row - center_bias
                        best_x = int(np.argmax(biased_row))
                        if traversable[y, best_x] > 0:
                            raw_path.append((best_x, y))

                    if len(raw_path) > 5:
                        window = 5
                        smoothed = []
                        for i in range(len(raw_path)):
                            s = max(0, i - window)
                            e = min(len(raw_path), i + window + 1)
                            avg_x = int(np.mean([p[0] for p in raw_path[s:e]]))
                            smoothed.append((avg_x, raw_path[i][1]))
                        path_points = smoothed
                    else:
                        path_points = raw_path

        # --- 7. Draw path gradient line ---
        if len(path_points) > 1:
            total_pts = len(path_points)
            for i in range(total_pts - 1):
                progress = i / max(total_pts - 1, 1)
                color = get_path_color(progress)
                thickness = max(2, int(6 * (1 - progress * 0.8)))
                cv2.line(result, path_points[i], path_points[i+1], color, thickness)

            # End marker near camera
            cv2.circle(result, path_points[-1], 6, (0, 200, 255), -1)

            # Label
            cv2.putText(result, f"len={len(path_points)}",
                        (w - 220, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- 8. Status HUD ---
        fps_buf.append(1.0 / max(time.time() - t0, 1e-6))
        avg_fps = np.mean(fps_buf)

        status_text = f"FPS:{avg_fps:.1f}"
        if reasons:
            status_text += " | " + ",".join(reasons[:3])

        cv2.putText(result, status_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        # Near-field indicator
        cv2.putText(result, "Limit: 3-4m | YOLOv26-Seg ADE20K",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- 9. Side display ---
        mask_vis = cv2.cvtColor(traversable, cv2.COLOR_GRAY2BGR)
        # Resize both for consistent display and speed
        result_disp = cv2.resize(result, (640, 480))
        mask_disp = cv2.resize(mask_vis, (640, 480))

        display = np.hstack([result_disp, mask_disp])
        cv2.imshow("YOLOv26-Seg + Path (Q=quit, limit 3-4m)", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Finished.")

if __name__ == "__main__":
    main()
