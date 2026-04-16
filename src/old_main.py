import os
from movement_commands import process_command, apply_esc_microsec
from picture_analyzer import capture_image_from_usb_camera, find_target_aruco, find_target_fallback, CAMERA_MATRIX, DISTORTION_COEFFS
import cv2
import time
import math

# === CONSTANTS ===

# DEBUG MODE
DEBUG_MODE = True

# Ensure OpenCV can access the display for imshow
os.environ["DISPLAY"] = ":0" # Ensure OpenCV can access the display for imshow

# IMAGE_FOLDER
IMAGE_FOLDER = f"captured_images/{int(time.time())}/"

# Camera
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# The camera is rotated 180° (bottom faces right when viewed from behind),
# so we rotate the frame 180° to correct it before any processing.
# So in this case LOGICAL_WIDTH = FRAME_WIDTH and LOGICAL_HEIGHT = FRAME_HEIGHT.
LOGICAL_WIDTH  = FRAME_WIDTH    # 1280
LOGICAL_HEIGHT = FRAME_HEIGHT    # 720

# ESC initialization
ESC_NEUTRAL      = 1000
ESC_START        = 1500
ESC_FULL_FORWARD = 2000
ESC_INIT_DELAY   = 1    # seconds between ESC init steps

# Decision making
NUM_FRAMES_FOR_DECISION = 2   # frames sampled per decision

# Pixel-band thresholds along the horizontal axis (after rotation).
# These divide the frame into 5 steering zones based on x position of target.
# 0 = left edge, LOGICAL_WIDTH = right edge.

ZONE_VERY_FAR_LEFT_MAX  = int(LOGICAL_WIDTH * 0.10)   #
ZONE_LEFT_MAX           = int(LOGICAL_WIDTH * 0.30)   #
ZONE_RIGHT_MIN          = int(LOGICAL_WIDTH * 0.70)   #
ZONE_VERY_FAR_RIGHT_MIN = int(LOGICAL_WIDTH * 0.90)   #

# Anything between LEFT_MAX and RIGHT_MIN is FORWARD
# "Close enough" threshold: target is near the bottom of the rotated frame.
# If the target was last seen here before disappearing, trigger final approach.

CLOSE_ENOUGH_Y_MIN = int(LOGICAL_HEIGHT * 0.80)   # bottom 20% of frame

# Steering durations (seconds)
TURN_DURATION_VERY_FAR = 0.3
TURN_DURATION_FAR      = 0.15

# Forward nudge duration after every steering decision (seconds)
FWD_NUDGE_DURATION = 0.3


# Forward advance duration after every steering decision (seconds)
FWD_NAVIGATION_DURATION = 0.3

# Search forward duration when target not found at all (seconds)
FWD_SEARCH_DURATION = 0.4

# Final approach forward duration before unloading (seconds)
FWD_FINAL_DURATION = 0.001

# === HELPERS ===
def rotate_frame(frame):

    """Rotate frame upside down to correct for camera orientation."""

    return cv2.rotate(frame, cv2.ROTATE_180)

def get_majority_target(cap):
    """
    Capture NUM_FRAMES_FOR_DECISION frames.

    1. Checks all for AprilTag. If >= 1 found, averages and returns.

    2. If 0 AprilTags, runs gradient fallback on the SAME frames.

       If >= 2 found, averages and returns.

    """
    frames = []
    aruco_detections = []
    # 1. Capture frames and look for ArUco
    for _ in range(NUM_FRAMES_FOR_DECISION):
        raw_frame = capture_image_from_usb_camera(cap)
        frame = rotate_frame(raw_frame)
        frames.append(frame)
        pos = find_target_aruco(frame)

        if pos is not None:
            pos = (int(pos[0]), int(pos[1]))
            aruco_detections.append(pos)

    last_frame = frames[-1] if frames else None

    # 2. Evaluate ArUco (Any hit is a success)
    if len(aruco_detections) > 0:
        avg_x = sum(pos[0] for pos in aruco_detections) // len(aruco_detections)
        avg_y = sum(pos[1] for pos in aruco_detections) // len(aruco_detections)

        # Draw on the actual frame the detections came from (Green for ArUco)
        if last_frame is not None:
            for pos in aruco_detections:
                cv2.circle(last_frame, pos, 10, (0, 255, 0), 2)

        # UNCOMMENT TO SAVE IMAGES FOR DEBUGGING (WILL CAUSE LAG)
        # os.makedirs(IMAGE_FOLDER, exist_ok=True)
        # cv2.imwrite(f"{IMAGE_FOLDER}/last_detections_{int(time.time())}.jpg", last_frame)

        return (avg_x, avg_y), last_frame

    # 3. Evaluate Fallback (Requires >= 2 hits)
    fallback_detections = []
    for frame in frames:
        pos = find_target_fallback(frame)
        if pos is not None:
            pos = (int(pos[0]), int(pos[1]))
            fallback_detections.append(pos)

    if len(fallback_detections) >= 2:
        avg_x = sum(pos[0] for pos in fallback_detections) // len(fallback_detections)
        avg_y = sum(pos[1] for pos in fallback_detections) // len(fallback_detections)

        # Draw on the actual frame the detections came from (Yellow for Fallback)
        if last_frame is not None:
            for pos in fallback_detections:
                cv2.circle(last_frame, pos, 10, (0, 255, 255), 2)

        # UNCOMMENT TO SAVE IMAGES FOR DEBUGGING (WILL CAUSE LAG)
        # os.makedirs(IMAGE_FOLDER, exist_ok=True)
        # cv2.imwrite(f"{IMAGE_FOLDER}/last_detections_{int(time.time())}.jpg", last_frame)

        return (avg_x, avg_y), last_frame

    return None, last_frame

def classify_position(x):
    """
    Map the target's x pixel coordinate to one of 5 steering states.
    Left/right are from the robot's perspective after frame rotation.
    """

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

def steer_by_state(state):
    """Issue turn command based on state, then stop."""

    if state == "VERY_FAR_LEFT":
        process_command("LEFT")
        time.sleep(TURN_DURATION_VERY_FAR)
        # process_command("STOP")

    elif state == "LEFT":
        process_command("LEFT")
        time.sleep(TURN_DURATION_FAR)
        # process_command("STOP")

    elif state == "VERY_FAR_RIGHT":
        process_command("RIGHT")
        time.sleep(TURN_DURATION_VERY_FAR)
        # process_command("STOP")

    elif state == "RIGHT":
        process_command("RIGHT")
        time.sleep(TURN_DURATION_FAR)
        # process_command("STOP")

    elif state == "FORWARD":
        process_command("FWD")
        time.sleep(FWD_NAVIGATION_DURATION)
        # process_command("STOP")

    else:
        print(f"Unknown state: {state}")

    # Always nudge forward after steering decision
    print("Nudging forward after steering decision...")
    process_command("FWD")
    time.sleep(FWD_NUDGE_DURATION)

def unload_charge():
    """Drop off the charge at the target. TODO: implement."""
    print("UNLOADING CHARGE... (placeholder)")
    pass

# === PHASES ===

def phase_init():
    """Start up the ESC."""
    print("Initializing ESC...")
    apply_esc_microsec(ESC_NEUTRAL)
    time.sleep(ESC_INIT_DELAY)
    # apply_esc_microsec(ESC_START)
    # time.sleep(ESC_INIT_DELAY)
    # apply_esc_microsec(ESC_FULL_FORWARD)
    # time.sleep(ESC_INIT_DELAY)

    print("ESC initialized.")


def phase_search(cap):
    """
    TODO: maybe fix?
    Look for the target. If not found, nudge forward and retry.
    Returns the first confirmed (x, y) position.
    """
    print("Searching for target...")
    while True:
        pos, _ = get_majority_target(cap)
        if pos is not None:
            print(f"Target found at {pos}")
            return pos

        print("Target not found — moving forward to search...")
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
            # Target lost — only trigger final approach if it was near the bottom
            if last_y is not None and last_y >= CLOSE_ENOUGH_Y_MIN:
                print(f"Target lost after being close (last y={last_y}). Proceeding to unload.")
                return

            else:
                # Lost too early — treat as search situation
                print("Target lost unexpectedly — searching forward...")
                process_command("FWD")
                time.sleep(FWD_SEARCH_DURATION)
                continue

        x, y     = pos
        last_y   = y
        state    = classify_position(x)
        print(f"Target at ({x}, {y}) → state: {state}")

        # # show the frame with detections and state for debugging (optional)
        # if frame is not None:
        #     # show the image with half of the resolution
        #     display_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        #     # write the state on the frame
        #     cv2.imshow(f"Navigation View: {state}", display_frame)
        #     cv2.waitKey(1)  # needed to update the imshow window

        steer_by_state(state)

        # Always move forward after command
        # process_command("FWD")
        # time.sleep(FWD_NAVIGATION_DURATION)
        # process_command("STOP")

def phase_final_approach():
    """Move forward for a fixed duration then unload."""
    print("Final approach...")
    process_command("FWD")
    time.sleep(FWD_FINAL_DURATION)
    process_command("STOP")
    print("Unloading charge...")
    unload_charge()
    print("Done!")
    # sleep for two seconds, then kill esc
    time.sleep(2)
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