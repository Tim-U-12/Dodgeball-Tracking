import cv2

def read_video(video_path):
  """Read a video file into a list of OpenCV frames."""
  cap = cv2.VideoCapture(video_path)
  frames = []
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frames.append(frame)
  return frames

def save_video(output_video_frames, output_video_path):
  """Save a list of OpenCV frames to an MP4 video file."""
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
  for frame in output_video_frames:
    out.write(frame)
  out.release()
