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

    def setPreferableBackend(self, backend: int) -> None:  # noqa: N802 - OpenCV style
        return None

    def setPreferableTarget(self, target: int) -> None:  # noqa: N802 - OpenCV style
        return None

    def forward(self, layer_names: list[str]) -> tuple[np.ndarray, np.ndarray]:  # noqa: N802
        scores = np.array([[[[self.score]]]], dtype=np.float32)
        geometry = np.zeros((1, 5, 1, 1), dtype=np.float32)
        return scores, geometry


def test_text_detector_high_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_net = _FakeNet(0.95)
    monkeypatch.setattr(cv2.dnn, "readNet", lambda model_path: fake_net)

    def fake_decode(self: TextDetector, scores, geometry, conf_thresh):
        return [(0.0, 0.0, 200.0, 200.0)], [0.9]

    monkeypatch.setattr(TextDetector, "_decode", fake_decode)
    monkeypatch.setattr(
        cv2.dnn,
        "NMSBoxes",
        lambda boxes, confidences, score, thresh: np.array([[0]]),
    )

    detector = TextDetector("dummy.pb")
    frame = np.zeros((320, 640, 3), dtype=np.uint8)

    has_text, confidence, total_ratio, largest_ratio = detector.detect(frame)

    assert has_text is True
    assert confidence == pytest.approx(0.95)
    assert total_ratio == pytest.approx((200 * 200) / (640 * 320))
    assert largest_ratio == pytest.approx((200 * 200) / (640 * 320))


def test_text_detector_filters_small_text_area(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_net = _FakeNet(0.5)
    monkeypatch.setattr(cv2.dnn, "readNet", lambda model_path: fake_net)

    def fake_decode(self: TextDetector, scores, geometry, conf_thresh):
        return [(10.0, 10.0, 15.0, 15.0)], [0.4]

    monkeypatch.setattr(TextDetector, "_decode", fake_decode)
    monkeypatch.setattr(
        cv2.dnn,
        "NMSBoxes",
        lambda boxes, confidences, score, thresh: np.array([[0]]),
    )

    detector = TextDetector("dummy.pb")
    frame = np.zeros((320, 640, 3), dtype=np.uint8)

    has_text, confidence, total_ratio, largest_ratio = detector.detect(frame)

    expected_area = (15 - 10) * (15 - 10)
    frame_area = 640 * 320

    assert has_text is False
    assert confidence == pytest.approx(0.5)
    assert total_ratio == pytest.approx(expected_area / frame_area)
    assert largest_ratio == pytest.approx(expected_area / frame_area)
