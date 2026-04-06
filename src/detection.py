import cv2
import numpy as np
from .config import *

def detect_balls(image, court_mask, ball_colour):
    court_only = cv2.bitwise_and(image, image, mask=court_mask)

    hsv = cv2.cvtColor(court_only, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ball_colour["lower"], ball_colour["upper"])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)

    if USE_MORPH_OPEN:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if USE_MORPH_CLOSE:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = cv2.bitwise_and(mask, mask, mask=court_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = w / float(h)
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            continue

        (xc, yc), radius = cv2.minEnclosingCircle(contour)
        center = (int(xc), int(yc))
        radius = int(radius)

        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            continue

        detections.append({
            "center": center,
            "radius": radius,
            "area": area,
            "bounding_box": (x, y, w, h),
            "circularity": circularity,
        })
        cv2.circle(result, center, radius, BALL_COLOUR_DRAW, BALL_OUTLINE_THICKNESS)
        cv2.circle(result, center, CENTER_DOT_RADIUS, CENTER_COLOUR, -1)

    return mask, result, detections
