import os
from movement_commands import process_command, apply_esc_microsec
from picture_analyzer import capture_image_from_usb_camera, find_target_aruco, find_target_fallback, CAMERA_MATRIX, DISTORTION_COEFFS
import cv2
import time
import math

#TODO: Look at code!!!

# === CONSTANTS ===

# Ensure OpenCV can access the display for imshow
os.environ["DISPLAY"] = ":0"

# Debug mode — set to False to disable image saving
DEBUG = True

# ENGINE mode 
ENGINE = True

# IMAGE_FOLDER
IMAGE_FOLDER = f"images/{int(time.time())}/"

# Camera
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# The camera is rotated 180° (bottom faces right when viewed from behind),
# so we rotate the frame 180° to correct it before any processing.
LOGICAL_WIDTH  = FRAME_WIDTH    # 1280
LOGICAL_HEIGHT = FRAME_HEIGHT   # 720

# ESC initialization
ESC_NEUTRAL      = 1000
ESC_START        = 1500
ESC_MIDDLE       = 1700
ESC_FULL_FORWARD = 2000
ESC_INIT_DELAY   = 1    # seconds between ESC init steps

# Decision making
NUM_FRAMES_FOR_DECISION = 1   # frames sampled per decision

# Pixel-band thresholds along the horizontal axis (after rotation).
ZONE_VERY_FAR_LEFT_MAX  = int(LOGICAL_WIDTH * 0.10)
ZONE_LEFT_MAX           = int(LOGICAL_WIDTH * 0.35)
ZONE_RIGHT_MIN          = int(LOGICAL_WIDTH * 0.65)
ZONE_VERY_FAR_RIGHT_MIN = int(LOGICAL_WIDTH * 0.90)

# "Close enough" threshold
CLOSE_ENOUGH_Y_MIN = int(LOGICAL_HEIGHT * 0.50)

# Steering durations (seconds)
TURN_RIGHT_DURATION_VERY_FAR = 0.2
TURN_RIGHT_DURATION_FAR      = 0.1

TURN_LEFT_DURATION_VERY_FAR  = 0.2
TURN_LEFT_DURATION_FAR       = 0.1

# Forward nudge / navigation / search durations (seconds)
FWD_NUDGE_DURATION       = 0.05
FWD_NAVIGATION_DURATION  = 0.1
FWD_SEARCH_DURATION      = 1
FWD_FINAL_DURATION       = 1

# === HELPERS ===

def get_forwarwd_time_final(y_pos):
    return   

def rotate_frame(frame):
    """Rotate frame upside down to correct for camera orientation."""
    return cv2.rotate(frame, cv2.ROTATE_180)


def save_decision_image(frame, pos, state):
    """
    Annotate *frame* with:
      - Vertical zone-boundary lines
      - The detected target position (circle + crosshair)
      - The chosen decision as large overlay text
      - A timestamp
    Then save to IMAGE_FOLDER.

    Parameters
    ----------
    frame : np.ndarray  – the (already rotated) camera frame
    pos   : tuple | None – (x, y) of the detected target, or None
    state : str          – the decision string, e.g. "LEFT", "FORWARD", …
    """
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    vis = frame.copy()
    h, w = vis.shape[:2]

    # --- Zone boundary lines (thin, semi-transparent white) ---
    zone_xs = [
        ZONE_VERY_FAR_LEFT_MAX,
        ZONE_LEFT_MAX,
        ZONE_RIGHT_MIN,
        ZONE_VERY_FAR_RIGHT_MIN,
    ]
    zone_labels = ["||", "|", "|", "||"]
    for x_pos, lbl in zip(zone_xs, zone_labels):
        cv2.line(vis, (x_pos, 0), (x_pos, h), (200, 200, 200), 1, cv2.LINE_AA)

    # --- Target marker ---
    if pos is not None:
        tx, ty = int(pos[0]), int(pos[1])
        # Outer circle
        cv2.circle(vis, (tx, ty), 18, (0, 255, 0), 2, cv2.LINE_AA)
        # Crosshair
        cv2.line(vis, (tx - 24, ty), (tx + 24, ty), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(vis, (tx, ty - 24), (tx, ty + 24), (0, 255, 0), 1, cv2.LINE_AA)
        # Coordinate label
        coord_txt = f"({tx}, {ty})"
        cv2.putText(vis, coord_txt, (tx + 22, ty - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    # --- Decision overlay ---
    # Map state → colour so it's easy to read at a glance
    state_colour = {
        "FORWARD":        (0,   255,   0),
        "LEFT":           (255, 165,   0),
        "VERY_FAR_LEFT":  (0,   100, 255),
        "RIGHT":          (255, 165,   0),
        "VERY_FAR_RIGHT": (0,   100, 255),
        "SEARCH_FWD":     (200, 200,   0),
    }.get(state, (255, 255, 255))

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.6
    thickness  = 3
    text       = f"DECISION: {state}"
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Dark background rectangle for legibility
    pad = 10
    cv2.rectangle(vis,
                  (pad, h - th - baseline - pad * 2),
                  (pad * 2 + tw, h - pad),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(vis, text,
                (pad * 2, h - baseline - pad),
                font, font_scale, state_colour, thickness, cv2.LINE_AA)

    # --- Timestamp ---
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(vis, ts, (w - 340, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

    # --- Save ---
    filename = os.path.join(IMAGE_FOLDER, f"decision_{int(time.time() * 1000)}.jpg")
    cv2.imwrite(filename, vis)
    print(f"  [img] saved → {filename}")


def get_majority_target(cap):
    """
    Capture NUM_FRAMES_FOR_DECISION frames.

    1. Checks all for AprilTag. If >= 1 found, averages and returns.
    2. If 0 AprilTags, runs gradient fallback on the SAME frames.
       If >= 2 found, averages and returns.
    """
    frames = []
    aruco_detections = []

    for _ in range(NUM_FRAMES_FOR_DECISION):
        raw_frame = capture_image_from_usb_camera(cap)
        frame = rotate_frame(raw_frame)
        frames.append(frame)
        pos = find_target_aruco(frame)
        if pos is not None:
            pos = (int(pos[0]), int(pos[1]))
            aruco_detections.append(pos)

    last_frame = frames[-1] if frames else None

    if len(aruco_detections) > 0:
        avg_x = sum(p[0] for p in aruco_detections) // len(aruco_detections)
        avg_y = sum(p[1] for p in aruco_detections) // len(aruco_detections)
        if last_frame is not None:
            for p in aruco_detections:
                cv2.circle(last_frame, p, 10, (0, 255, 0), 2)
        return (avg_x, avg_y), last_frame

    fallback_detections = []
    for frame in frames:
        pos = find_target_fallback(frame)
        if pos is not None:
            pos = (int(pos[0]), int(pos[1]))
            fallback_detections.append(pos)

    if len(fallback_detections) >= 2:
        avg_x = sum(p[0] for p in fallback_detections) // len(fallback_detections)
        avg_y = sum(p[1] for p in fallback_detections) // len(fallback_detections)
        if last_frame is not None:
            for p in fallback_detections:
                cv2.circle(last_frame, p, 10, (0, 255, 255), 2)
        return (avg_x, avg_y), last_frame

    return None, last_frame


def classify_position(x):
    """Map the target's x pixel coordinate to one of 5 steering states."""
    if x < ZONE_VERY_FAR_LEFT_MAX:
        return "VERY_FAR_LEFT"
    elif x < ZONE_LEFT_MAX:
        return "LEFT"
    elif x > ZONE_VERY_FAR_RIGHT_MIN:
        return "VERY_FAR_RIGHT"
    elif x > ZONE_RIGHT_MIN:
        return "RIGHT"
    else:
        return "FORWARD"


def steer_by_state(state, frame=None, pos=None):
    """
    Stop the car, save an annotated decision image, then issue the
    appropriate movement command.

    Parameters
    ----------
    state : str          – one of the classify_position() return values
    frame : np.ndarray   – camera frame to annotate (may be None)
    pos   : tuple | None – (x, y) target position for annotation
    """
    # --- Persist the decision visually ---
    process_command("STOP")
    if DEBUG and frame is not None:
        save_decision_image(frame, None, state)

    # --- Execute the movement ---
    if state == "VERY_FAR_LEFT":
        process_command("LEFT")
        time.sleep(TURN_LEFT_DURATION_VERY_FAR)

    elif state == "LEFT":
        process_command("LEFT")
        time.sleep(TURN_LEFT_DURATION_FAR)

    elif state == "VERY_FAR_RIGHT":
        process_command("RIGHT")
        time.sleep(TURN_RIGHT_DURATION_VERY_FAR)

    elif state == "RIGHT":
        process_command("RIGHT")
        time.sleep(TURN_RIGHT_DURATION_FAR)

    elif state == "FORWARD":
        process_command("FWD")
        time.sleep(FWD_NAVIGATION_DURATION)

    else:
        print(f"Unknown state: {state}")


def unload_charge():
    """Drop off the charge at the target. TODO: implement."""
    process_command("ARM_DOWN")
    time.sleep(0.5)
    process_command("ARM_STOP")


# === PHASES ===

def phase_init():
    """Start up the ESC."""
    print("Initializing ESC...")
    apply_esc_microsec(ESC_NEUTRAL)
    time.sleep(ESC_INIT_DELAY)

    if ENGINE:
        apply_esc_microsec(ESC_START)
        time.sleep(ESC_INIT_DELAY)
        apply_esc_microsec(ESC_MIDDLE)
        time.sleep(ESC_INIT_DELAY)
        apply_esc_microsec(ESC_FULL_FORWARD)
        time.sleep(ESC_INIT_DELAY)
        print("ESC initialized.")
    else:
        print("ENGINE mode disabled — skipping ESC initialization.")


def phase_search(cap):
    """
    Look for the target. If not found, nudge forward and retry.
    Returns the first confirmed (x, y) position.
    """
    print("Searching for target...")
    while True:
        pos, frame = get_majority_target(cap)
        if pos is not None:
            print(f"Target found at {pos}")
            return pos

        print("Target not found — moving forward to search...")
        # Stop + save image showing the 'no target' search decision
        process_command("STOP")
        if frame is not None:
            save_decision_image(frame, None, "SEARCH_FWD")
        process_command("FWD")
        time.sleep(FWD_SEARCH_DURATION)
        process_command("STOP")


def phase_navigate(cap):
    """
    Steer toward the target using pixel-zone classification.
    Exits when the target disappears after having been near the bottom of the frame.
    """
    print("Navigating toward target...")
    last_y = None

    while True:
        pos, frame = get_majority_target(cap)

        if pos is None:
            if last_y is not None and last_y >= CLOSE_ENOUGH_Y_MIN:
                print(f"Target lost after being close (last y={last_y}). Proceeding to unload.")
                if DEBUG and frame is not None:
                    save_decision_image(frame, None, "ENTERING_FINAL_APPROACH")
                return
            else:
                print("Target lost unexpectedly — searching forward...")
                process_command("STOP")
                if DEBUG and frame is not None:
                    save_decision_image(frame, None, "SEARCH_FWD")
                process_command("FWD")
                time.sleep(FWD_SEARCH_DURATION)
                continue

        x, y   = pos
        last_y = y
        state  = classify_position(x)
        print(f"Target at ({x}, {y}) → state: {state}")

        # Stop + save + move
        steer_by_state(state, frame=frame, pos=pos)


def phase_final_approach():
    """Move forward for a fixed duration then unload."""
    print("Final approach...")
    process_command("STOP")
    time.sleep(1)
    process_command("FWD")
    time.sleep(FWD_FINAL_DURATION)
    process_command("STOP")
    print("Unloading charge...")
    # unload_charge()
    time.sleep(2)
    process_command("FWD")
    time.sleep(0.1)
    process_command("STOP")
    time.sleep(2)
    print("Done!")
    apply_esc_microsec(ESC_NEUTRAL)


# === MAIN ===
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Failed to open USB camera")

    try:
        phase_init()
        phase_search(cap)
        phase_navigate(cap)
        phase_final_approach()

    finally:
        process_command("STOP")
        apply_esc_microsec(ESC_NEUTRAL)
        cap.release()
        cv2.destroyAllWindows()