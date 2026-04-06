import cv2
import numpy as np
from .config import *

def select_court_points(image):
    display = image.copy()
    court_points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal display, court_points

        if event == cv2.EVENT_LBUTTONDOWN:
            court_points.append((x, y))

            cv2.circle(display, (x, y), POINT_RADIUS, POINT_COLOUR, -1)

            if len(court_points) > 1:
                cv2.line(display, court_points[-2], court_points[-1], LINE_COLOUR, LINE_THICKNESS)

            cv2.imshow(SELECT_WINDOW_NAME, display)

    print("Click around the court boundary.")
    print("Press ENTER when done, or press 'r' to reset.")

    cv2.namedWindow(SELECT_WINDOW_NAME)
    cv2.setMouseCallback(SELECT_WINDOW_NAME, mouse_callback)

    while True:
        cv2.imshow(SELECT_WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            court_points = []
            display = image.copy()

        elif key == 13:
            if len(court_points) > 2:
                cv2.line(display, court_points[-1], court_points[0], LINE_COLOUR, LINE_THICKNESS)
                cv2.imshow(SELECT_WINDOW_NAME, display)
                cv2.waitKey(200)
            break

    cv2.destroyWindow(SELECT_WINDOW_NAME)

    if len(court_points) < 3:
        raise ValueError("You need at least 3 points to define the court.")

    return court_points


def create_court_mask(image, court_points):
    court_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(court_points, dtype=np.int32)
    cv2.fillPoly(court_mask, [pts], 255)
    return court_mask