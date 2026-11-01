"""Run via `python -m uvpc` or the `uvpc` console command."""

import argparse
from pathlib import Path
import sys
from time import perf_counter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimental YOLO person detection + IoU tracking")
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0) or local video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO COCO .pt file (default may auto-download)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection threshold (0, 1]")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="Tracking IoU threshold (0, 1]")
    parser.add_argument("--max-missing", type=int, default=10, help="Missed frames to retain a track")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0: unlimited)")
    parser.add_argument("--output", help="Optional annotated .mp4 path")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI preview for headless runs")
    parser.add_argument("--mirror", action="store_true", help="Mirror input horizontally (webcam preview)")
    return parser


def run(args: argparse.Namespace) -> int:
    import cv2

    from .camera import VideoSource
    from .detection import PersonDetector
    from .tracking import IoUTracker
    from .visualization import draw_tracks

    if args.max_frames < 0:
        raise ValueError("--max-frames must be nonnegative")
    if args.output and Path(args.output).suffix.lower() != ".mp4":
        raise ValueError("--output must be an .mp4 file")

    source = int(args.source) if args.source.isdecimal() else args.source
    tracker = IoUTracker(args.iou_threshold, args.max_missing)
    detector = PersonDetector(args.model, args.confidence)
    writer = None
    frame_count = 0
    start = perf_counter()
    try:
        with VideoSource(source) as capture:
            while True:
                frame = capture.read()
                if frame is None:
                    break
                if args.mirror:
                    frame = cv2.flip(frame, 1)
                tracks = tracker.update(detector.detect(frame))
                frame_count += 1
                fps = frame_count / max(perf_counter() - start, 1e-9)
                annotated = draw_tracks(frame, tracks, fps)
                if args.output:
                    if writer is None:
                        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                        height, width = annotated.shape[:2]
                        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 capture.fps, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not create output video: {args.output}")
                    writer.write(annotated)
                if not args.no_display:
                    cv2.imshow("UVPC Vision Tracker (q to quit)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if args.max_frames and frame_count >= args.max_frames:
                    break
    finally:
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames in {perf_counter() - start:.2f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
