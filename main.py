import cv2
from src.read_frame import read_image
from src.config import BALL_COLOUR
from src.court import select_court_points, create_court_mask
from src.detection import detect_balls

IMAGE_PATH = "data/raw/frames/BUNDHA-yellow-2.png"

COURT_MASK_WINDOW_NAME = "Court Mask"
YELLOW_MASK_WINDOW_NAME = "Yellow Mask In Court"
DETECTED_BALLS_WINDOW_NAME = "Detected Balls"

def main():
    image = read_image(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    court_points = select_court_points(image)
    court_mask = create_court_mask(image, court_points)

    mask, result, detections = detect_balls(image, court_mask, BALL_COLOUR)

    print(f"Detected {len(detections)} ball(s)")
    for index, detection in enumerate(detections, start=1):
        print(
            f"Ball {index}: "
            f"center={detection['center']}, "
            f"radius={detection['radius']}, "
            f"area={detection['area']:.2f}, "
            f"bounding_box={detection['bounding_box']}, "
            f"circularity={detection['circularity']:.2f}"
        )

    cv2.imshow(COURT_MASK_WINDOW_NAME, court_mask)
    cv2.imshow(YELLOW_MASK_WINDOW_NAME, mask)
    cv2.imshow(DETECTED_BALLS_WINDOW_NAME, result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()