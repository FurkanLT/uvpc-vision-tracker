# UVPC Vision Tracker

[![Pure Python checks](https://github.com/FurkanLT/uvpc-vision-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/FurkanLT/uvpc-vision-tracker/actions/workflows/ci.yml)

**Experimental Computer Vision & Person Tracking Project** — a small student portfolio project, not a production-ready system.

## Overview

UVPC explores a practical webcam/video pipeline: YOLO person detections (COCO class 0), bounding-box geometry, simple frame-to-frame association, and annotated video. The emphasis is on a small, understandable learning project.

## Features

- OpenCV webcam index or local video file input.
- Person bounding boxes from a real Ultralytics YOLO model (no custom detector training claimed).
- **Basic IoU tracking** with temporary per-session IDs, configurable association threshold and missed-frame budget.
- Bounding-box area, intersection, IoU, overlap, clipping and coordinate scaling utilities.
- Visual overlays for boxes, IDs, confidence and approximate processing FPS.
- Optional MP4 export and headless processing.
- CPU-only unit tests for the pure geometry and tracker logic; no camera or GPU required.

**Not implemented:** cross-camera person Re-ID, face recognition/expression classification, depth estimation, 3D reconstruction, AGI, or any production deployment.

## Tech Stack

Python 3.10+, OpenCV, [Ultralytics YOLO](https://docs.ultralytics.com/), pytest (tests). YOLO models run through their own PyTorch/runtime dependencies. The tracking and geometry modules use only the Python standard library.

## Architecture

~~~text
src/uvpc/
  camera.py          # OpenCV VideoCapture wrapper
  detection.py       # YOLO COCO person inference
  geometry.py        # Validated (x1, y1, x2, y2) operations
  tracking.py        # Greedy IoU association, temporary IDs
  similarity.py      # Tested cosine utilities (not a Re-ID model)
  visualization.py   # Bounding boxes and FPS overlay
  __main__.py        # CLI and video loop
examples/geometry_demo.py  # Scripted tracker example; not detector validation
tests/                     # Hardware-independent unit tests
~~~

## Installation

~~~bash
python -m venv .venv
# Activate .venv (Linux/macOS: source .venv/bin/activate; Windows: .venv\Scripts\activate)
python -m pip install -e '.[test]'
~~~

The first run with the default yolov8n.pt can download pretrained YOLO weights from Ultralytics; network access may be required. Alternatively, download an appropriate COCO person-detection model yourself and provide its local path with --model. Model weights and datasets are intentionally **not** tracked. Review the applicable upstream model/software licenses before redistributing.

## Running

~~~bash
# Webcam index 0 (press q to exit)
python -m uvpc --source 0

# A local video, no GUI, annotated output
python -m uvpc --source path/to/video.mp4 --no-display --output outputs/annotated.mp4

# Local YOLO weights and ten-frame sample
python -m uvpc --source path/to/video.mp4 --model path/to/yolov8n.pt --max-frames 10

# Show all options (no heavy runtime imports or weights needed)
python -m uvpc --help
~~~

For an explicit local file, pass a file path. A source consisting only of digits is treated as a camera index. --mirror is an optional horizontal flip; it is off by default. GUI display requires a desktop environment; use --no-display on headless systems. Processing FPS includes per-frame work but excludes model initialization, so it is not a benchmark.

## Testing

~~~bash
python -m compileall -q src tests examples
python -m pytest -q
python examples/geometry_demo.py
~~~

The synthetic example feeds **scripted boxes**, not an image detector, to demonstrate ID association. Unit tests never validate YOLO accuracy. Real inference and live-camera compatibility require a separately run integration test with suitable hardware/model files. The CI workflow deliberately avoids downloading large model dependencies and only tests the dependency-free core.

## Limitations

- **Not production-ready**, not benchmarked for accuracy, latency, robustness or privacy compliance.
- IoU matching is greedy: IDs may switch during occlusion, crossings, fast motion and camera changes. IDs are not human identities. **There is no Re-ID model or fallback masquerading as Re-ID.**
- YOLO person performance depends on external weights, input conditions, package compatibility and hardware. Runtime downloads require connectivity.
- Camera indices, codecs, FPS reporting and GUI support vary across operating systems and devices.
- No depth sensor support, face emotion labels or 2D-to-3D inference; these features are outside the project scope.
- Images/video can contain sensitive information; obtain consent and avoid publishing footage of others. The demo does not stream or upload video, but optional MP4 outputs remain on disk.

## Why I Built It

I built this as a learning project to explore real-time Computer Vision pipelines, YOLO inference, basic person tracking, camera handling, and geometry. The cleanup intentionally favors a small, auditable implementation over unsupported or exaggerated capabilities.
