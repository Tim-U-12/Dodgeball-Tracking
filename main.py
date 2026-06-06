from trackers import Tracker


def main():
    # initialise Tracker
    tracker = Tracker('models/best.pt')

    # Track, annotate, and save the video without loading every frame into memory.
    tracker.process_video(
        input_video_path='data/raw/videos/dodgeball.mp4',
        output_video_path='output_videos/output_video.mp4',
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl',
    )

if __name__ == "__main__":
    main()
