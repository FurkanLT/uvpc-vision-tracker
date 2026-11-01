"""YOLO COCO person detection (class 0); weights are NOT bundled."""

from pathlib import Path

from .geometry import area, clip
from .tracking import Detection


class PersonDetector:
    def __init__(self, model: str = "yolov8n.pt", confidence: float = 0.5):
        if not 0 < confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        if model != "yolov8n.pt" and not Path(model).is_file():
            raise FileNotFoundError(f"Model file not found: {model}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required; install with 'pip install -e .'") from exc
        # The default 'yolov8n.pt' may be downloaded by Ultralytics on first run.
        self.model = YOLO(model)
        self.confidence = confidence

    def detect(self, frame) -> list[Detection]:
        height, width = frame.shape[:2]
        results = self.model.predict(source=frame, classes=[0], conf=self.confidence, verbose=False)
        detections: list[Detection] = []
        for box in results[0].boxes:
            coords = tuple(float(v) for v in box.xyxy[0].tolist())
            bounded = clip(coords, width, height)
            if area(bounded) > 0:
                detections.append(Detection(bounded, float(box.conf[0])))
        return detections
