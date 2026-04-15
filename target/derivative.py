import numpy as np
import cv2

def find_target(image: np.ndarray, threshold = 70) -> tuple[float, float]:
    # ------------------------------------------------------------------ #
    # 0. Normalise to float so derivative arithmetic is lossless          #
    # ------------------------------------------------------------------ #
    img = image.astype(np.float64)

    # If the image is colour, convert to grayscale first
    if img.ndim == 3:
        # Standard luminance weights (ITU-R BT.601)
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    # ------------------------------------------------------------------ #
    # 1. Left (backward) derivative along the x-axis (columns)           #
    #    d[r, c] = img[r, c] - img[r, c-1]                               #
    #    For c == 0 the left neighbour is undefined; we use 0 (padding).  #
    # ------------------------------------------------------------------ #
    derivative = np.zeros_like(img)
    derivative[:, 1:] = abs(img[:, 1:] - img[:, :-1])   # columns 1 … W-1
    # derivative[:, 0] stays 0  (left-border padding)

    # ------------------------------------------------------------------ #
    # 2. Threshold                                                         #
    # ------------------------------------------------------------------ #
    binary = np.where(derivative > threshold, 255.0, 0.0)

    # ------------------------------------------------------------------ #
    # 3. Weighted centroid of white pixels                                 #
    # ------------------------------------------------------------------ #
    # np.argwhere returns [[row0, col0], [row1, col1], ...]
    white_pixels = np.argwhere(binary == 255)

    if white_pixels.size == 0:
        return None  # No target found

    # argwhere gives (row, col) → map to (y, x)
    y_coords = white_pixels[:, 0]
    x_coords = white_pixels[:, 1]
    
    # Use derivative values as weights, scaled with a polynomial function
    weights = np.power(derivative[y_coords, x_coords], 8)

    x_mean = float(np.average(x_coords, weights=weights))
    y_mean = float(np.average(y_coords, weights=weights))

    return (x_mean, y_mean)

# ------------------------------------------------------------------ #
# Quick self-test                                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    for i in range(42, 66):
        try:
            # Create a simple 5x5 grayscale test image with a vertical edge at column 3
            IMAGE_NAME = f"last_detections_17761970{i}.jpg"
            IMAGE_PATH = f"images/v4/{IMAGE_NAME}"
            pos = find_target(cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE), threshold=20)

            # show the orignal image with the detected position marked
            img = cv2.imread(IMAGE_PATH)
            if pos is not None:
                cv2.circle(img, (int(pos[0]), int(pos[1])), 10, (0, 255, 0), 2)
            cv2.imshow('Detected Target', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error processing {IMAGE_NAME}: {e}")