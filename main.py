import cv2
from src.read_frame import read_image

from src.utils import (
    BALL_COLOUR,
    create_court_mask,
    detect_balls,
    select_court_points,
)


def main():
    image = read_image("data/raw/frames/BUNDHA-yellow-2.png")

    if image is None:
        raise FileNotFoundError("Could not load image: data/raw/frames/BUNDHA-yellow-2.png")

    court_points = select_court_points(image)
    court_mask = create_court_mask(image, court_points)

    mask, result, detections = detect_balls(image, court_mask, BALL_COLOUR)

    print(f"Detected {len(detections)} ball(s)")
    for i, detection in enumerate(detections, start=1):
        print(f"Ball {i}: center={detection['center']}, radius={detection['radius']}")

    cv2.imshow("Court Mask", court_mask)
    cv2.imshow("Yellow Mask In Court", mask)
    cv2.imshow("Detected Balls", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()