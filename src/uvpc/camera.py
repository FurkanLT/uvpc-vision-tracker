"""Minimal OpenCV webcam/video capture wrapper."""

import math


class VideoSource:
    def __init__(self, source: int | str):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required; install with 'pip install -e .'") from exc
        self._is_camera = isinstance(source, int)
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Cannot open video source: {source!r}")
        reported_fps = self._capture.get(cv2.CAP_PROP_FPS)
        self.fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else 30.0

    def read(self):
        ok, frame = self._capture.read()
        if not ok:
            if self._is_camera:
                raise RuntimeError("Webcam frame read failed")
            return None  # end of video file
        return frame

    def close(self):
        self._capture.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
