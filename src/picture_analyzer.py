import cv2
import numpy as np
import math

# --- CALIBRATION DATA ---
CAMERA_MATRIX = np.array([
    [830.66800186,   0.        , 334.94647384],
    [  0.        , 831.11105661, 258.04204427],
    [  0.        ,   0.        ,   1.        ]
])

DISTORTION_COEFFS = np.array([
    [-2.34969634e-01,  5.20219557e-02, -6.73304245e-05,
     -2.11012015e-03, -5.14734027e-01]
])

# --- ARUCO SETUP ---
try:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
except AttributeError:
    ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_16h5)

try:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()
    DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
except AttributeError:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
    DETECTOR = None


# --- TARGET DETECTION ALGORITHMS ---

def find_target_aruco(image):
    """Primary: Strict geometric decoding for ID 0."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if DETECTOR is not None:
        corners, ids, rejected = DETECTOR.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    if ids is None or len(corners) == 0:
        return None

    for i in range(len(ids)):
        if ids[i][0] == 0:
            marker_corners = corners[i][0]
            cx = int(np.mean(marker_corners[:, 0]))
            cy = int(np.mean(marker_corners[:, 1]))
            return (cx, cy)

    return None


def find_target_fallback(image: np.ndarray, threshold=70):
    """Fallback: Gradient derivative centroid tracking."""
    img = image.astype(np.float64)

    # Convert to grayscale
    if img.ndim == 3:
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    # Calculate left (backward) derivative along the x-axis
    derivative = np.zeros_like(img)
    derivative[:, 1:] = abs(img[:, 1:] - img[:, :-1])

    # Threshold
    binary = np.where(derivative > threshold, 255.0, 0.0)
    white_pixels = np.argwhere(binary == 255)

    if white_pixels.size == 0:
        return None

    y_coords = white_pixels[:, 0]
    x_coords = white_pixels[:, 1]
    
    # Weights scaled with polynomial function
    weights = np.power(derivative[y_coords, x_coords], 8)

    # Calculate weighted mean
    x_mean = np.average(x_coords, weights=weights)
    y_mean = np.average(y_coords, weights=weights)

    # Cast to int so it plays nicely with cv2 drawing functions downstream
    return (int(x_mean), int(y_mean))


def find_target(image):
    """
    Unified router: Tries AprilTag first. If it fails, falls back to the gradient detector.
    """
    # 1. Try the primary Aruco decoder
    target = find_target_aruco(image)
    
    if target is not None:
        # Optional: Print to console so you know which algorithm is triggering
        # print("Target acquired via AprilTag") 
        return target
        
    # 2. If Aruco fails, try the fallback gradient algorithm
    target = find_target_fallback(image)
    
    if target is not None:
        # print("Target acquired via Gradient Fallback")
        return target
        
    # 3. Both failed
    return None


# --- CAMERA STREAM ---

def capture_image_from_usb_camera(cap):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture image from camera")
    return frame


def process_usb_camera_stream(cap, max_frames=None):
    frame_count = 0

    while True:
        raw_frame = capture_image_from_usb_camera(cap)
        
        # Flatten the image
        frame = cv2.undistort(raw_frame, CAMERA_MATRIX, DISTORTION_COEFFS)

        # Call the unified router function
        target_pos = find_target(frame)
        
        if target_pos:
            print(f"Frame {frame_count} | Target Found at: {target_pos}")
        else:
            print(f"Frame {frame_count} | Target lost")

        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        raise RuntimeError("Failed to open USB camera")

    try:
        print("Starting stream analysis...")
        process_usb_camera_stream(cap, max_frames=50)
    finally:
        cap.release()
        cv2.destroyAllWindows()