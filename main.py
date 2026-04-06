import cv2
from src.read_frame import read_video
from src.config import BALL_COLOUR
from src.court import select_court_points, create_court_mask
from src.detection import detect_balls

VIDEO_PATH = "data/raw/videos/test.mp4"

COURT_MASK_WINDOW_NAME = "Court Mask"
YELLOW_MASK_WINDOW_NAME = "Yellow Mask In Court"
DETECTED_BALLS_WINDOW_NAME = "Detected Balls"

def main():
    cap = read_video(VIDEO_PATH)

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise FileNotFoundError(f"Could not read first frame from video: {VIDEO_PATH}")

    court_points = select_court_points(first_frame)
    court_mask = create_court_mask(first_frame, court_points)

    frame_count = 0

    while True:
        if frame_count == 0:
            frame = first_frame
        else:
            ret, frame = cap.read()
            if not ret:
                break

        mask, result, detections = detect_balls(frame, court_mask, BALL_COLOUR)

        print(f"Frame {frame_count}: Detected {len(detections)} ball(s)")
        for index, detection in enumerate(detections, start=1):
            print(
                f"  Ball {index}: "
                f"center={detection['center']}, "
                f"radius={detection['radius']}, "
                f"area={detection['area']:.2f}, "
                f"bounding_box={detection['bounding_box']}, "
                f"circularity={detection['circularity']:.2f}"
            )

        cv2.imshow(COURT_MASK_WINDOW_NAME, court_mask)
        cv2.imshow(YELLOW_MASK_WINDOW_NAME, mask)
        cv2.imshow(DETECTED_BALLS_WINDOW_NAME, result)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            cv2.waitKey(0)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()