"""Simple, deterministic IoU association; NOT person re-identification."""

from dataclasses import dataclass
from math import isfinite

from .geometry import Box, area, iou


@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float

    def __post_init__(self) -> None:
        area(self.box)  # also validates coordinates
        if not (isfinite(self.confidence) and 0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    box: Box
    confidence: float


@dataclass
class _TrackState:
    box: Box
    missed: int = 0


class IoUTracker:
    """Greedy one-to-one matching between consecutive frame detections.

    IDs are session-local, not identity claims; occlusion/crossing may switch IDs.
    A track expires after more than `max_missing` consecutive missing frames.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 10):
        if not 0 < iou_threshold <= 1:
            raise ValueError("iou_threshold must be in (0, 1]")
        if not isinstance(max_missing, int) or max_missing < 0:
            raise ValueError("max_missing must be a nonnegative integer")
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self._tracks: dict[int, _TrackState] = {}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        candidates = []
        for track_id, previous in self._tracks.items():
            for index, detection in enumerate(detections):
                score = iou(previous.box, detection.box)
                if score >= self.iou_threshold:
                    candidates.append((-score, track_id, index))
        candidates.sort()  # highest IoU first, deterministic ties

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        output: dict[int, TrackedDetection] = {}
        for _, track_id, index in candidates:
            if track_id in matched_tracks or index in matched_detections:
                continue
            detection = detections[index]
            self._tracks[track_id] = _TrackState(detection.box)
            matched_tracks.add(track_id)
            matched_detections.add(index)
            output[index] = TrackedDetection(track_id, detection.box, detection.confidence)

        for track_id in list(self._tracks):
            if track_id not in matched_tracks:
                state = self._tracks[track_id]
                state.missed += 1
                if state.missed > self.max_missing:
                    del self._tracks[track_id]

        for index, detection in enumerate(detections):
            if index not in matched_detections:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _TrackState(detection.box)
                output[index] = TrackedDetection(track_id, detection.box, detection.confidence)

        return [output[index] for index in range(len(detections))]
