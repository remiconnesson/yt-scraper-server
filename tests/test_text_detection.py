import cv2
import numpy as np
import pytest

from slides_extractor.extract_slides.text_detection import TextDetector


class _FakeNet:
    def __init__(self, score: float) -> None:
        self.last_input = None
        self.score = score

    def setInput(self, blob: np.ndarray) -> None:  # noqa: N802 - OpenCV style
        self.last_input = blob

    def forward(self, layer_names: list[str]) -> tuple[np.ndarray, np.ndarray]:  # noqa: N802
        scores = np.array([[[[self.score]]]], dtype=np.float32)
        geometry = np.zeros((1, 5, 1, 1), dtype=np.float32)
        return scores, geometry


def test_text_detector_high_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_net = _FakeNet(0.95)
    monkeypatch.setattr(cv2.dnn, "readNet", lambda model_path: fake_net)

    detector = TextDetector("dummy.pb")
    frame = np.zeros((320, 640, 3), dtype=np.uint8)

    has_text, confidence = detector.detect(frame)

    assert has_text is True
    assert confidence == pytest.approx(0.95)
