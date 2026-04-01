import cv2
import numpy as np


BALL_COLOUR = {
    "lower": np.array([18, 100, 100]),
    "upper": np.array([40, 255, 255]),
}


def select_court_points(image):
    display = image.copy()
    court_points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal display, court_points

        if event == cv2.EVENT_LBUTTONDOWN:
            court_points.append((x, y))

            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

            if len(court_points) > 1:
                cv2.line(display, court_points[-2], court_points[-1], (255, 0, 0), 2)

            cv2.imshow("Select Court", display)

    print("Click around the court boundary.")
    print("Press ENTER when done, or press 'r' to reset.")

    cv2.namedWindow("Select Court")
    cv2.setMouseCallback("Select Court", mouse_callback)

    while True:
        cv2.imshow("Select Court", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            court_points = []
            display = image.copy()

        elif key == 13:  # Enter
            break

    cv2.destroyWindow("Select Court")

    if len(court_points) < 3:
        raise ValueError("You need at least 3 points to define the court.")

    return court_points


def create_court_mask(image, court_points):
    court_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(court_points, dtype=np.int32)
    cv2.fillPoly(court_mask, [pts], 255)
    return court_mask


def detect_balls(image, court_mask, ball_colour):
    court_only = cv2.bitwise_and(image, image, mask=court_mask)

    hsv = cv2.cvtColor(court_only, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ball_colour["lower"], ball_colour["upper"])

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = cv2.bitwise_and(mask, mask, mask=court_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20 or area > 500:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            result,
            "Ball",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return mask, result