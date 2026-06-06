from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
sys.path.append("../")
from utils import get_centre_of_bbox, get_bbox_width

class Tracker:
  def __init__(self, model_path):
    self.model = YOLO(model_path)
    self.tracker = sv.ByteTrack()

  def get_frame_tracks(self, detection):
    """Convert one YOLO detection result into tracked player and ball dictionaries."""
    cls_names = detection.names
    cls_names_inv = {v:k for k,v in cls_names.items()}

    # Convert to supervision Dection format
    detection_supervision = sv.Detections.from_ultralytics(detection)

    # Track objects
    detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

    frame_tracks = {
      "balls": {},
      "players": {},
    }

    for frame_detection in detection_with_tracks:
      bbox = frame_detection[0].tolist()
      cls_id = frame_detection[3]
      track_id = frame_detection[4]

      if cls_id == cls_names_inv['player'] and track_id != -1:
        frame_tracks["players"][track_id] = {"bbox": bbox}

    for ball_id, frame_detection in enumerate(detection_supervision):
      bbox = frame_detection[0].tolist()
      cls_id = frame_detection[3]

      if cls_id == cls_names_inv['ball']:
        frame_tracks["balls"][ball_id] = {"bbox": bbox}

    return frame_tracks

  def detect_frames(self, frames):
    batch_size = 20
    detections = []
    for i in range(0, len(frames), batch_size):
      detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
      detections += detections_batch
    return detections

  def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
    
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
      with open(stub_path, 'rb') as f:
        tracks = pickle.load(f)
      if len(tracks["players"]) == len(frames) and len(tracks["balls"]) == len(frames):
        return tracks
      print("Ignoring stale track stub because it does not match the video length.")
    
    detections = self.detect_frames(frames)
    tracks = {
      "balls":[],
      "players":[],
    }

    for frame_num, detection in enumerate(detections):
      frame_tracks = self.get_frame_tracks(detection)
      tracks["balls"].append(frame_tracks["balls"])
      tracks["players"].append(frame_tracks["players"])

    if stub_path is not None:
      with open(stub_path, 'wb') as f:
        pickle.dump(tracks,f)
    return tracks

  def process_video(
      self,
      input_video_path,
      output_video_path,
      read_from_stub=False,
      stub_path=None,
      batch_size=8,
      max_frames=None,
    ):
    """Track and annotate a video in small batches so frames are not all kept in memory."""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
      raise ValueError(f"Could not open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cached_tracks = None
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
      with open(stub_path, 'rb') as f:
        cached_tracks = pickle.load(f)
      if len(cached_tracks["players"]) != total_frames or len(cached_tracks["balls"]) != total_frames:
        print("Ignoring stale track stub because it does not match the video length.")
        cached_tracks = None

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
      cap.release()
      raise ValueError(f"Could not create output video: {output_video_path}")

    generated_tracks = {
      "balls": [],
      "players": [],
    }
    frame_num = 0

    try:
      while True:
        frames = []
        for _ in range(batch_size):
          if max_frames is not None and frame_num + len(frames) >= max_frames:
            break

          ret, frame = cap.read()
          if not ret:
            break
          frames.append(frame)

        if not frames:
          break

        if cached_tracks is not None:
          tracks = {
            "balls": cached_tracks["balls"][frame_num:frame_num + len(frames)],
            "players": cached_tracks["players"][frame_num:frame_num + len(frames)],
          }
        else:
          detections = self.model.predict(frames, conf=0.1)
          tracks = {
            "balls": [],
            "players": [],
          }

          for detection in detections:
            frame_tracks = self.get_frame_tracks(detection)
            tracks["balls"].append(frame_tracks["balls"])
            tracks["players"].append(frame_tracks["players"])

          generated_tracks["balls"].extend(tracks["balls"])
          generated_tracks["players"].extend(tracks["players"])

        output_frames = self.draw_annotations(frames, tracks)
        for output_frame in output_frames:
          out.write(output_frame)

        frame_num += len(frames)
        if frame_num % 100 == 0:
          print(f"Processed {frame_num}/{total_frames} frames")

        if max_frames is not None and frame_num >= max_frames:
          break
    finally:
      cap.release()
      out.release()

    if cached_tracks is None and stub_path is not None and max_frames is None:
      os.makedirs(os.path.dirname(stub_path), exist_ok=True)
      with open(stub_path, 'wb') as f:
        pickle.dump(generated_tracks, f)

    return output_video_path
  
  def draw_ellipse(self, frame, bbox, color, track_id):
    y2 = int(bbox[3])
    x_centre, _ = get_centre_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(
      frame,
      center=(x_centre, y2),
      axes=(int(.75*width), int(0.5*width)),
      angle=0.0,
      startAngle=20,
      endAngle=160,
      color=color,
      thickness=2,
      lineType=cv2.LINE_4,
    )
    return frame

  def draw_ball_marker(self, frame, bbox, color):
    """Draw a simple circle around a detected ball."""
    x_centre, y_centre = get_centre_of_bbox(bbox)
    radius = max(6, int(get_bbox_width(bbox)))

    cv2.circle(
      frame,
      center=(x_centre, y_centre),
      radius=radius,
      color=color,
      thickness=2,
      lineType=cv2.LINE_4,
    )
    return frame


  def draw_annotations(self, video_frames, tracks):
    output_video_frames = []
    for frame_num, frame in enumerate(video_frames):
      frame = frame.copy()

      player_dict = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
      ball_dict = tracks["balls"][frame_num] if frame_num < len(tracks["balls"]) else {}

      # draw players
      for track_id, player in player_dict.items():
        frame = self.draw_ellipse(frame, player["bbox"], (0,0,255), track_id)

      # draw balls
      for _, ball in ball_dict.items():
        frame = self.draw_ball_marker(frame, ball["bbox"], (0,255,255))
    
      output_video_frames.append(frame)
    
    return output_video_frames
