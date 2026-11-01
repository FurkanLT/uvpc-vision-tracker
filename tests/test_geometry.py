import pytest

from uvpc.geometry import area, clip, denormalize, intersection_area, iou, normalize, overlap_fraction
from uvpc.similarity import cosine_distance, cosine_similarity


def test_area_and_intersection():
    assert area((0, 0, 10, 20)) == 200
    assert area((2, 2, 2, 9)) == 0
    assert intersection_area((0, 0, 10, 10), (5, 5, 15, 15)) == 25


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0
    assert iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0


def test_overlap_is_directional():
    assert overlap_fraction((0, 0, 10, 10), (0, 0, 5, 5)) == 0.25
    assert overlap_fraction((0, 0, 5, 5), (0, 0, 10, 10)) == 1
    assert overlap_fraction((0, 0, 0, 0), (0, 0, 10, 10)) == 0


def test_normalize_roundtrip():
    original = (20, 10, 80, 50)
    assert normalize(original, 100, 200) == (0.2, 0.05, 0.8, 0.25)
    assert denormalize(normalize(original, 100, 200), 100, 200) == pytest.approx(original)


def test_clip_uses_half_open_image_boundaries():
    assert clip((-5, -2, 110, 210), 100, 200) == (0, 0, 100, 200)
    assert clip((120, 220, 140, 240), 100, 200) == (100, 200, 100, 200)


@pytest.mark.parametrize("box", [(2, 0, 1, 1), (0, 2, 1, 1), (0, 0, float("nan"), 1)])
def test_bad_boxes_rejected(box):
    with pytest.raises(ValueError):
        area(box)


@pytest.mark.parametrize("size", [(0, 10), (10, -1), (float("inf"), 10)])
def test_invalid_dimensions(size):
    with pytest.raises(ValueError):
        normalize((0, 0, 1, 1), *size)


def test_cosine_helpers():
    assert cosine_similarity([1, 0], [1, 0]) == 1
    assert cosine_similarity([1, 0], [0, 1]) == 0
    assert cosine_similarity([1, 0], [-1, 0]) == -1
    assert cosine_distance([1, 0], [1, 0]) == 0
    assert cosine_distance([1, 0], [-1, 0]) == 2


@pytest.mark.parametrize("a,b", [([0, 0], [1, 2]), ([1], [1, 2]), ([], []), ([float("nan")], [1])])
def test_bad_cosine_inputs(a, b):
    with pytest.raises(ValueError):
        cosine_similarity(a, b)
