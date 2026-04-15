import cv2
import numpy as np
    
def find_target(image):
    # 1. Convert to grayscale
    gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Noise reduction: blur to smooth out texture
    gray_im = cv2.GaussianBlur(gray_im, (5, 5), 0)

    # 3. Threshold: pixels darker than 70 are marked as foreground
    _, thresh = cv2.threshold(gray_im, 70, 255, cv2.THRESH_BINARY_INV)

    # 4. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 5. Find the leftmost contour (smallest x in bounding box)
    leftmost_contour = min(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

    # 6. Compute center of the bounding box
    x, y, w, h = cv2.boundingRect(leftmost_contour)
    cx = x + w // 2
    cy = y + h // 2

    return (cx, cy)


for i in range(3, 10):
    FILENAME = f"{i}blocks.jpg"
    IMAGE_PATH = f"images/v3/{FILENAME}"

    image = cv2.imread(IMAGE_PATH)
    target_pos = find_target(image) 


    if target_pos:
        cx, cy = target_pos
        
        # Draw the green crosshairs at the target center
        cv2.circle(image, (cx, cy), 10, (0, 255, 0), 2)
        cv2.circle(image, (cx, cy), 2, (0, 0, 255), 3)
        
    else:
        cv2.putText(image, "Target Lost", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the live video window (Requires VNC)
    cv2.imshow("Vision Test", image)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed