from ultralytics import YOLO

model = YOLO('yolo26m')

results = model.predict('./data/raw/videos/dodgeball.mp4', save=True)
print(results[0])
print('===================================')
for box in results[0].boxes:
    print(box)