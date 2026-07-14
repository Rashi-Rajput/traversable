"""
Real-time traversable path detection (city & terrain).
Uses YOLO26n-seg + classical CV for maximum accuracy.
Shows contour + contour mask only. No output save.
"""

import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# VIDEO PATH — hardcoded once, nowhere else
# ============================================================
VIDEO_PATH = "input.mp4"

# ============================================================
# Model / Performance Settings
# ============================================================
MODEL_NAME = "yolo26n-seg.pt"
YOLO_IMGSZ = 320
YOLO_CONF = 0.30
YOLO_IOU = 0.40
YOLO_FRAME_SKIP = 3

PROC_WIDTH = 640

# ============================================================
# Obstacle Classes (COCO)
# ============================================================
OBSTACLE_CLASSES = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 26, 28, 56, 57, 58, 59, 60, 61, 62, 63,
    65, 67, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
}


def load_yolo():
    """Load YOLO26n-seg on CPU."""
    print("Loading YOLO26n-seg on CPU...")
    model = YOLO(MODEL_NAME)
    model.to("cpu")

    dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
    try:
        model.predict(dummy, imgsz=YOLO_IMGSZ, verbose=False, device="cpu")
    except Exception:
        pass
    print("Model ready.")
    return model


def get_obstacle_mask(model, frame):
    """Get obstacle mask from YOLO segmentation."""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    try:
        results = model.predict(
            frame,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            device="cpu",
            verbose=False,
        )
    except Exception:
        return mask

    if not results or results[0].masks is None:
        return mask

    r = results[0]
    masks_data = r.masks.data.cpu().numpy()
    boxes_data = r.boxes.data.cpu().numpy()

    for i in range(len(masks_data)):
        cls_id = int(boxes_data[i][5])
        conf = float(boxes_data[i][4])

        if conf < YOLO_CONF or cls_id not in OBSTACLE_CLASSES:
            continue

        m = cv2.resize(masks_data[i], (w, h), interpolation=cv2.INTER_LINEAR)
        mask[m > 0.5] = 255

    if cv2.countNonZero(mask) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def compute_floor_score(image, prev_score):
    """
    Classical CV floor/traversability scoring.
    Optimized for near-field (closer to camera).
    """
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # ROI: focus on near-field (bottom 55%)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[int(h * 0.45):, :] = 255

    # Edge density (avoid edges = better for traversal)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray_blur, 75, 150)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    edge_density = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)

    # Texture analysis
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    texture = cv2.absdiff(gray, blur)
    texture = cv2.blur(texture.astype(np.float32), (11, 11))
    texture = cv2.normalize(texture, None, 0, 1, cv2.NORM_MINMAX)

    # Color similarity to road/floor (sample from bottom center)
    patch = lab[h - 50:h - 10, w // 2 - 40:w // 2 + 40]
    if patch.size > 0:
        floor_color = np.mean(patch.reshape(-1, 3), axis=0)
    else:
        floor_color = np.array([128, 128, 128], dtype=np.float32)

    color_dist = np.linalg.norm(lab.astype(np.float32) - floor_color, axis=2)
    color_dist = cv2.normalize(color_dist, None, 0, 1, cv2.NORM_MINMAX)
    color_score = 1 - color_dist

    # Combined score
    score = (
        0.35 * color_score
        + 0.45 * (1 - edge_density)
        + 0.20 * (1 - texture)
    )
    score = cv2.GaussianBlur(score, (9, 9), 0)

    # Temporal smoothing
    if prev_score is not None:
        score = 0.85 * score + 0.15 * prev_score

    return score, roi_mask


def get_traversable_mask(score, roi_mask):
    """Extract traversable region from score map."""
    threshold = np.mean(score) * 1.08
    floor_mask = ((score > threshold) * 255).astype(np.uint8)
    floor_mask = cv2.bitwise_and(floor_mask, roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only region connected to bottom (near camera)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(floor_mask, 4)

    traversable = np.zeros_like(floor_mask)
    if num_labels > 1:
        bottom_y = floor_mask.shape[0] - 10
        for lbl in range(1, num_labels):
            if labels[bottom_y, floor_mask.shape[1] // 2] == lbl:
                traversable[labels == lbl] = 255
                break

        if cv2.countNonZero(traversable) == 0:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                best = 1 + int(np.argmax(areas))
                traversable[labels == best] = 255

    return traversable


def draw_contours(frame, traversable_mask, obstacle_mask, fps):
    """Draw only contours - no center line."""
    result = frame.copy()
    h, w = frame.shape[:2]

    # Draw traversable contour (green)
    trav_contours, _ = cv2.findContours(
        traversable_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if trav_contours:
        cv2.drawContours(result, trav_contours, -1, (0, 255, 0), 3)

    # Draw obstacle contours (red)
    obs_contours, _ = cv2.findContours(
        obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if obs_contours:
        cv2.drawContours(result, obs_contours, -1, (0, 0, 255), 2)

    # FPS
    cv2.putText(
        result,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return result


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        sys.exit(1)

    model = load_yolo()

    # Get video dimensions
    ret, frame = cap.read()
    if not ret:
        print("Empty video")
        sys.exit(1)

    h0, w0 = frame.shape[:2]
    proc_w = PROC_WIDTH
    proc_h = int(h0 * (proc_w / w0))
    if proc_h % 2:
        proc_h += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prev_score = None
    cached_obstacle = np.zeros((proc_h, proc_w), dtype=np.uint8)
    frame_idx = 0
    fps_history = []

    print("Running... press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t0 = time.perf_counter()

        proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        # Classical CV floor detection
        score, roi_mask = compute_floor_score(proc, prev_score)
        prev_score = score.copy()

        # YOLO obstacle detection (every N frames)
        if frame_idx == 1 or frame_idx % YOLO_FRAME_SKIP == 1:
            cached_obstacle = get_obstacle_mask(model, proc)

        # Combine: floor - obstacles
        traversable = get_traversable_mask(score, roi_mask)
        traversable = cv2.bitwise_and(traversable, cv2.bitwise_not(cached_obstacle))

        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        traversable = cv2.morphologyEx(traversable, cv2.MORPH_CLOSE, kernel, iterations=1)

        # FPS
        elapsed = time.perf_counter() - t0
        cur_fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_history.append(cur_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = float(np.mean(fps_history))

        # Draw contours only
        output = draw_contours(proc, traversable, cached_obstacle, avg_fps)

        # Side-by-side: original with contours + traversable mask
        mask_vis = cv2.cvtColor(traversable, cv2.COLOR_GRAY2BGR)
        display = np.hstack([
            cv2.resize(output, (640, 480)),
            cv2.resize(mask_vis, (640, 480)),
        ])

        cv2.imshow("Traversable Path - Contour View", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if fps_history:
        print(f"\nAvg FPS: {np.mean(fps_history):.1f}")


if __name__ == "__main__":
    main()
