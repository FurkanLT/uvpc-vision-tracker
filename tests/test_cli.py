import sys
from types import ModuleType

from uvpc.__main__ import build_parser, main


def test_help_without_opencv_or_yolo(capsys):
    with __import__("pytest").raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    assert "--source" in capsys.readouterr().out


def test_missing_model_fails_cleanly(capsys, monkeypatch):
    # Avoid importing OpenCV: missing-model checking happens before using it.
    monkeypatch.setitem(sys.modules, "cv2", ModuleType("cv2"))
    assert main(["--model", "/definitely/not/a/model.pt", "--no-display"]) == 1
    assert "Model file not found" in capsys.readouterr().err
