import cv2
import numpy as np

# Something that would be cool to add later would be a box where the color detection is and
# have it change colors when fire is detected there, like from green to red or something


# Convert the input image to the HSV color space
def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# Create a binary mask where the pixels within the specified color range are set to 1, others set to 0
def create_mask(hsv_image, lower_bound, upper_bound):
    return cv2.inRange(hsv_image, lower_bound, upper_bound)


# Find the contours/boundaries of connected components in the binary mask
def find_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# Extract regions of interest (ROIs) from the original image using the contours
def extract_rois(frame, contour):
    if contour is None:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    return frame[y:y+h, x:x+w]


# Main function for detecting possible fire regions in image
def detect_fire_region(frame, lower_bound = np.array([0, 74, 200]), upper_bound = np.array([18, 166, 230])):
    hsv_image = convert_to_hsv(frame)  # Convert to HSV
    mask = create_mask(hsv_image, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    largest_contour = find_contour(mask)  # Find largest contour
    fire_region = extract_rois(frame, largest_contour)  # Extract ROIs

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        return fire_region, x, y, largest_contour
    else:
        return None, None, None, None


# Visualizes detected regions, drawing a rectangle around them
def visualize_detection(frame: object, contour):
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame
