from utils import read_video, save_video

def main():
  # read video
  video_frames = read_video('data/raw/videos/dodgeball.mp4')

  # save video
  save_video(video_frames, 'output_videos/output_video.mp4')

if __name__ == "__main__":
  main()
