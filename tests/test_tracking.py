import pytest

from uvpc.tracking import Detection, IoUTracker


def det(x=0, score=0.9):
    return Detection((x, 0, x + 10, 10), score)


def test_persistent_id_with_overlap():
    tracker = IoUTracker(iou_threshold=0.5)
    assert tracker.update([det()])[0].track_id == 1
    assert tracker.update([det(1)])[0].track_id == 1


def test_distinct_people_and_one_to_one_matching():
    tracker = IoUTracker(iou_threshold=0.2)
    assert [v.track_id for v in tracker.update([det(0), det(30)])] == [1, 2]
    assert [v.track_id for v in tracker.update([det(31), det(1)])] == [2, 1]
    assert [v.track_id for v in tracker.update([det(1), det(2)])] == [1, 3]


def test_timeout_and_new_identity_after_loss():
    tracker = IoUTracker(max_missing=1)
    assert tracker.update([det()])[0].track_id == 1
    assert tracker.update([]) == []
    assert tracker.update([det(1)])[0].track_id == 1
    tracker.update([])
    tracker.update([])
    assert tracker.update([det()])[0].track_id == 2


def test_zero_missing_expires_immediately():
    tracker = IoUTracker(max_missing=0)
    tracker.update([det()])
    tracker.update([])
    assert tracker.update([det()])[0].track_id == 2


@pytest.mark.parametrize("threshold,missed", [(0, 2), (1.1, 2), (0.3, -1)])
def test_invalid_tracker_parameters(threshold, missed):
    with pytest.raises(ValueError):
        IoUTracker(threshold, missed)


def test_invalid_detection():
    with pytest.raises(ValueError):
        Detection((0, 0, 1, 1), float("nan"))
