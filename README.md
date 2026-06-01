# Dodgeball-Tracking
This application is used to track the positions of dodgeballs on court

## Dataset

The Ultralytics YOLO object detection dataset is configured in
`data/dodgeball/data.yaml`. It contains a contiguous split of the labelled
video frames:

- `260` training images: frames `000001` to `000260`
- `65` validation images: frames `000261` to `000325`
- Class `0`: `ball`
- Class `1`: `player`

Train from the repository root with:

```bash
yolo detect train data=data/dodgeball/data.yaml model=yolo26n.pt
```

The original images and CVAT export are retained temporarily in
`data/labelled-backup` and `data/test1-cvat-backup`.
