"""Pixel-coordinate bounding boxes in (x1, y1, x2, y2) format.

Coordinates use half-open bounds: the right/bottom edges may equal the image
width/height. A zero-area box is valid; reversed or non-finite boxes are not.
"""

from __future__ import annotations

from math import isfinite
from typing import TypeAlias

Box: TypeAlias = tuple[float, float, float, float]


def _check_box(box: Box) -> None:
    if len(box) != 4 or not all(isfinite(v) for v in box):
        raise ValueError("A box must contain four finite coordinates")
    if box[2] < box[0] or box[3] < box[1]:
        raise ValueError("Box corners are reversed")


def _check_size(width: float, height: float) -> None:
    if not (isfinite(width) and isfinite(height) and width > 0 and height > 0):
        raise ValueError("Image width and height must be finite and positive")


def area(box: Box) -> float:
    _check_box(box)
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_area(a: Box, b: Box) -> float:
    _check_box(a)
    _check_box(b)
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def iou(a: Box, b: Box) -> float:
    """Intersection over union; returns 0 for two empty boxes."""
    overlap = intersection_area(a, b)
    union = area(a) + area(b) - overlap
    return overlap / union if union > 0 else 0.0


def overlap_fraction(a: Box, b: Box) -> float:
    """Fraction of the *first* box covered by the second (not symmetric)."""
    denominator = area(a)
    return intersection_area(a, b) / denominator if denominator > 0 else 0.0


def clip(box: Box, width: float, height: float) -> Box:
    _check_box(box)
    _check_size(width, height)
    x1, y1, x2, y2 = box
    return (
        min(width, max(0.0, x1)),
        min(height, max(0.0, y1)),
        min(width, max(0.0, x2)),
        min(height, max(0.0, y2)),
    )


def normalize(box: Box, width: float, height: float) -> Box:
    """Scale pixel coordinates; out-of-frame boxes may exceed [0, 1]."""
    _check_box(box)
    _check_size(width, height)
    return (box[0] / width, box[1] / height, box[2] / width, box[3] / height)


def denormalize(box: Box, width: float, height: float) -> Box:
    _check_box(box)
    _check_size(width, height)
    return (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
