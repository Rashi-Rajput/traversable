import cv2
import numpy as np

VIDEO_PATH = "input.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

prev_score = None

def get_path_color(progress):
    r = int(255 * (1 - progress))
    g = int(255 * (1 - progress))
    b = int(255 * progress)
    return (b, g, r)

while True:
    ret, image = cap.read()
    if not ret:
        break

    h, w = image.shape[:2]

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[int(h * 0.45):, :] = 255

    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray_blur, 80, 160)
    edges = cv2.GaussianBlur(edges, (5,5), 0)
    edge_density = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    texture = cv2.absdiff(gray, blur)
    texture = cv2.blur(texture.astype(np.float32), (15, 15))
    texture = cv2.normalize(texture, None, 0, 1, cv2.NORM_MINMAX)

    patch = lab[h-60:h-10, w//2-30:w//2+30]

    if patch.size > 0:
        floor_color = np.mean(patch.reshape(-1, 3), axis=0)
    else:
        floor_color = np.array([128, 128, 128], dtype=np.float32)

    color_dist = np.linalg.norm(lab.astype(np.float32) - floor_color, axis=2)
    color_dist = cv2.normalize(color_dist, None, 0, 1, cv2.NORM_MINMAX)
    color_score = 1 - color_dist

    score = 0.30 * color_score + 0.50 * (1 - edge_density) + 0.20 * (1 - texture)
    score = cv2.GaussianBlur(score, (11, 11), 0)

    if prev_score is not None:
        score = 0.90 * score + 0.10 * prev_score

    prev_score = score.copy()

    threshold = np.mean(score) * 1.05
    floor_mask = (score > threshold).astype(np.uint8) * 255
    floor_mask = cv2.bitwise_and(floor_mask, roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(floor_mask, 4)

    seed_x = w // 2
    seed_y = h - 20

    traversable = np.zeros_like(floor_mask)

    if labels[seed_y, seed_x] > 0:
        traversable[labels == labels[seed_y, seed_x]] = 255

    result = image.copy()

   
    contours, _ = cv2.findContours(traversable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    path_points = []

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 800:
            cv2.drawContours(result, [largest], -1, (0, 255, 0), 3)

            dist_map = cv2.distanceTransform(traversable, cv2.DIST_L2, 3)

            y_coords = np.where(traversable > 0)[0]
            if len(y_coords) > 0:
                min_y = np.min(y_coords)

                raw_path = []

                for y in range(seed_y, min_y, -5):
                    row = dist_map[y, :]

                    if np.max(row) > 0:
                        center_bias = np.abs(np.arange(w) - w // 2) * 0.05
                        biased_row = row - center_bias

                        best_x = np.argmax(biased_row)

                        if traversable[y, best_x] > 0:
                            raw_path.append((best_x, y))

                if len(raw_path) > 5:
                    path_points = []
                    window = 5

                    for i in range(len(raw_path)):
                        start_idx = max(0, i - window)
                        end_idx = min(len(raw_path), i + window)

                        avg_x = int(np.mean([p[0] for p in raw_path[start_idx:end_idx]]))
                        path_points.append((avg_x, raw_path[i][1]))

 
    if len(path_points) > 1:
        total_pts = len(path_points)

        for i in range(total_pts - 1):
            pt1 = path_points[i]
            pt2 = path_points[i + 1]

            progress = i / total_pts
            color = get_path_color(progress)
            thickness = max(2, int(6 * (1 - progress)))

            cv2.line(result, pt1, pt2, color, thickness)

        cv2.putText(result, f"path_len={len(path_points)}",
                    (w - 200, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

   
    mask_vis = cv2.cvtColor(traversable, cv2.COLOR_GRAY2BGR)

    display = np.hstack([
        cv2.resize(result, (640, 480)),
        cv2.resize(mask_vis, (640, 480))
    ])

    cv2.imshow("Traversable path (clean)", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
