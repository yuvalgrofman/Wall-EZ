import os
import cv2
from movement_commands import apply_esc_microsec

# Import the zones and our unified target finder from main
from main import (
    ZONE_VERY_FAR_LEFT_MAX,
    ZONE_LEFT_MAX,
    ZONE_RIGHT_MIN,
    ZONE_VERY_FAR_RIGHT_MIN,
    get_majority_target
)

# define the display
os.environ["DISPLAY"] = ":0"
apply_esc_microsec(1000)  # Neutral for ESC

# create a camera object
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

try:
    while True:
        # get_majority_target does all the heavy lifting (captures, rotates, finds, and draws circles)
        # It now returns BOTH the position and the image!
        pos, frame = get_majority_target(cap)

        if pos is not None:
            print(f"Detected target at {pos}")
        else:
            # add text to the image that we couldn't detect target
            cv2.putText(frame, "No target detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("No target detected in this 5-frame cycle")

        # draw vertical lines to show the steering zones
        height, width = frame.shape[:2]
        cv2.line(frame, (ZONE_VERY_FAR_LEFT_MAX, 0), (ZONE_VERY_FAR_LEFT_MAX, height), (255, 0, 0), 2)
        cv2.line(frame, (ZONE_LEFT_MAX, 0), (ZONE_LEFT_MAX, height), (255, 0, 0), 2)
        cv2.line(frame, (ZONE_RIGHT_MIN, 0), (ZONE_RIGHT_MIN, height), (255, 0, 0), 2)
        cv2.line(frame, (ZONE_VERY_FAR_RIGHT_MIN, 0), (ZONE_VERY_FAR_RIGHT_MIN, height), (255, 0, 0), 2)

        # Shrink the frame by 50% purely for the VNC viewer
        # 1280x720 becomes 640x360
        display_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # show the image
        cv2.imshow("Camera Check", display_frame)

        # if the user presses 'q', exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()