import cv2
import numpy as np


# HSV colour range used to detect yellow dodgeballs in the image.
BALL_COLOUR = {
    "lower": np.array([18, 100, 100]),
    "upper": np.array([40, 255, 255]),
}


def select_court_points(image):
    """
    Let the user manually select the court boundary by clicking points
    around the court in the displayed image.

    Controls:
    - Left mouse click: add a boundary point
    - r: reset all selected points
    - Enter: finish selecting points

    Args:
        image: The original image as a NumPy array.

    Returns:
        A list of (x, y) tuples representing the selected court boundary.

    Raises:
        ValueError: If fewer than 3 points are selected.
    """
    display = image.copy()
    court_points = []

    def mouse_callback(event, x, y, flags, param):
        """
        Handle mouse click events for selecting court boundary points.

        Args:
            event: OpenCV mouse event type.
            x: X-coordinate of the mouse click.
            y: Y-coordinate of the mouse click.
            flags: Additional event flags.
            param: Extra parameters passed by OpenCV.
        """
        nonlocal display, court_points

        if event == cv2.EVENT_LBUTTONDOWN:
            court_points.append((x, y))

            # Draw the selected point.
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

            # Draw a line from the previous point to the new point.
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
            # Reset the selected points and restore the original image.
            court_points = []
            display = image.copy()

        elif key == 13:  # Enter key
            break

    cv2.destroyWindow("Select Court")

    if len(court_points) < 3:
        raise ValueError("You need at least 3 points to define the court.")

    return court_points


def create_court_mask(image, court_points):
    """
    Create a binary mask for the selected court area.

    The polygon formed by the selected court points is filled in white,
    while everything outside the court remains black.

    Args:
        image: The original image as a NumPy array.
        court_points: A list of (x, y) tuples defining the court boundary.

    Returns:
        A binary mask where the court region is white (255)
        and the background is black (0).
    """
    court_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(court_points, dtype=np.int32)
    cv2.fillPoly(court_mask, [pts], 255)
    return court_mask


def detect_balls(image, court_mask, ball_colour):
    """
    Detect yellow balls inside the selected court region.

    The function:
    1. Applies the court mask to ignore everything outside the court
    2. Converts the image to HSV colour space
    3. Thresholds the image to isolate yellow areas
    4. Cleans the mask using morphological operations
    5. Finds contours that match expected ball size and shape
    6. Draws bounding boxes and labels around detected balls

    Args:
        image: The original image as a NumPy array.
        court_mask: A binary mask of the court region.
        ball_colour: A dictionary containing HSV lower and upper bounds.

    Returns:
        A tuple containing:
        - mask: The processed binary mask showing yellow regions in the court
        - result: A copy of the original image with detected balls outlined
    """
    # Keep only the pixels inside the selected court.
    court_only = cv2.bitwise_and(image, image, mask=court_mask)

    # Convert to HSV for more reliable colour detection.
    hsv = cv2.cvtColor(court_only, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ball_colour["lower"], ball_colour["upper"])

    # Remove small noise and close small gaps.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply the court mask again to guarantee detections stay inside the court.
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

        # Circularity helps filter shapes that are not ball-like.
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