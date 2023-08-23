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
def find_contours(mask):
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


# Extract regions of interest (ROIs) from the original image using the contours
def extract_rois(frame, contours):
    rois = [cv2.boundingRect(contour) for contour in contours]
    return [frame[y:y+h, x:x+w] for (x, y, w, h) in rois]


# Main function for detecting possible fire regions in image
def detect_fire_regions(frame, lower_bound, upper_bound):
    hsv_image = convert_to_hsv(frame)  # Convert to HSV
    mask = create_mask(hsv_image, lower_bound, upper_bound)  # Create color mask
    contours = find_contours(mask)  # Find contours
    return extract_rois(frame, contours)  # Extract ROIs