import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def process_image_diff(image_path, offset=5):
    # Load the image in grayscale (black and white)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Could not find or open the image.")
        return

    # Create two versions of the image:
    # 1. The original image excluding the last 'offset' pixels horizontally
    # 2. The image shifted by 'offset' pixels, excluding the first 'offset' pixels
    img_left = img[:, :-offset].astype(np.int16)
    img_right = img[:, offset:].astype(np.int16)

    # Subtract and take absolute value
    # We use int16 to avoid overflow during subtraction before taking absolute value
    diff_img = np.abs(img_left - img_right).astype(np.uint8)

    # Save the resulting image
    # output_filename = 'processed_result.jpg'
    # cv2.imwrite(output_filename, diff_img)
    # print(f"Processed image saved as {output_filename}")

    # Optional: Display the result with cv2.imshow (commented out for debugging mode)
    cv2.imshow('Difference Image', diff_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return diff_img

def gray_image_threshold(gray_image, threshold=128):
    # Load the image in grayscale
    if gray_image is None:
        print("Error: Could not find or open the image.")
        return

    # Apply thresholding
    _, binary_img = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # Save the resulting image
    # output_filename = 'thresholded_result.jpg'
    # cv2.imwrite(output_filename, img)
    # print(f"Thrjsholded image saved as {output_filename}")

    # Optional: Display the result with cv2.imshow (commented out for debugging mode)
    cv2.imshow('Thresholded Image', binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage (assuming 'image.jpg' is in the same directory)
if __name__ == "__main__":
    for i in range(3, 10):
        IMAGE_NAME = f"{i}blocks.jpg"
        IMAGE_PATH = f"images/v3/{IMAGE_NAME}"
        diff_image = process_image_diff(IMAGE_PATH, offset=i)
        # Please perform histogram equalization on the diff_image before thresholding
        gray_image_threshold(diff_image, threshold=70)