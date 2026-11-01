"""Draw detections and basic telemetry onto frames."""

from .tracking import TrackedDetection


def draw_tracks(frame, tracks: list[TrackedDetection], fps: float):
    import cv2

    for track in tracks:
        x1, y1, x2, y2 = (round(v) for v in track.box)
        color = ((track.track_id * 71) % 256, (track.track_id * 131) % 256, 40)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track.track_id}  {track.confidence:.2f}",
                    (x1, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, f"Detections: {len(tracks)}  FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame
