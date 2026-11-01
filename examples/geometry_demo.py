"""Model-free smoke demo of bbox matching; NOT detector accuracy validation."""

from uvpc.tracking import Detection, IoUTracker

tracker = IoUTracker(iou_threshold=0.2)
for number, box in enumerate([(10, 10, 50, 90), (14, 12, 54, 92), (18, 14, 58, 94)], 1):
    tracks = tracker.update([Detection(box, confidence=0.9)])
    print(f"Synthetic frame {number}: temporary track ID {tracks[0].track_id}")
