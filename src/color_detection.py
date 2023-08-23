import cv2
import numpy as np


# Convert the input image to the HSV color space
def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# Create a binary mask where the pixels within the specified color range are set to 1, others set to 0
def create_mask(hsv_image, lower_bound, upper_bound):
    return cv2.inRange(hsv_image, lower_bound, upper_bound)
