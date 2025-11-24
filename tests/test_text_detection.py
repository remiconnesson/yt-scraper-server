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


def test_text_detector_counts_central_box(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_net = _FakeNet(0.95)
    monkeypatch.setattr(cv2.dnn, "readNet", lambda model_path: fake_net)

    def fake_decode(self: TextDetector, scores, geometry, conf_thresh):
        return [(100.0, 60.0, 540.0, 260.0)], [0.9]

    monkeypatch.setattr(TextDetector, "_decode", fake_decode)
    monkeypatch.setattr(
        cv2.dnn,
        "NMSBoxes",
        lambda boxes, confidences, score, thresh: np.array([[0]]),
    )

    detector = TextDetector("dummy.pb")
    frame = np.zeros((320, 640, 3), dtype=np.uint8)

    has_text, confidence, total_ratio, largest_ratio, boxes = detector.detect(frame)

    expected_area = (540 - 100) * (260 - 60)
    frame_area = 640 * 320

    assert has_text is True
    assert confidence == pytest.approx(0.95)
    assert total_ratio == pytest.approx(expected_area / frame_area)
    assert largest_ratio == pytest.approx(expected_area / frame_area)
    assert boxes == [(100, 60, 540, 260)]


def test_text_detector_ignores_corner_logo(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_net = _FakeNet(0.6)
    monkeypatch.setattr(cv2.dnn, "readNet", lambda model_path: fake_net)

    def fake_decode(self: TextDetector, scores, geometry, conf_thresh):
        return [(600.0, 10.0, 620.0, 30.0)], [0.55]

    monkeypatch.setattr(TextDetector, "_decode", fake_decode)
    monkeypatch.setattr(
        cv2.dnn,
        "NMSBoxes",
        lambda boxes, confidences, score, thresh: np.array([[0]]),
    )

    detector = TextDetector("dummy.pb")
    frame = np.zeros((320, 640, 3), dtype=np.uint8)

    has_text, confidence, total_ratio, largest_ratio, boxes = detector.detect(frame)

    assert has_text is False
    assert confidence == pytest.approx(0.6)
    assert total_ratio == 0.0
    assert largest_ratio == 0.0
    assert boxes == [(600, 10, 620, 30)]


def test_text_detector_only_counts_central_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_net = _FakeNet(0.8)
    monkeypatch.setattr(cv2.dnn, "readNet", lambda model_path: fake_net)

    def fake_decode(self: TextDetector, scores, geometry, conf_thresh):
        return [
            (5.0, 5.0, 30.0, 30.0),
            (200.0, 100.0, 420.0, 180.0),
        ], [0.4, 0.7]

    monkeypatch.setattr(TextDetector, "_decode", fake_decode)
    monkeypatch.setattr(
        cv2.dnn,
        "NMSBoxes",
        lambda boxes, confidences, score, thresh: np.array([[0], [1]]),
    )

    detector = TextDetector("dummy.pb")
    frame = np.zeros((320, 640, 3), dtype=np.uint8)

    has_text, confidence, total_ratio, largest_ratio, boxes = detector.detect(frame)

    expected_center_area = (420 - 200) * (180 - 100)
    frame_area = 640 * 320

    assert has_text is True
    assert confidence == pytest.approx(0.8)
    assert total_ratio == pytest.approx(expected_center_area / frame_area)
    assert largest_ratio == pytest.approx(expected_center_area / frame_area)
    assert boxes == [(5, 5, 30, 30), (200, 100, 420, 180)]


def test_text_detector_returns_false_without_central_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_net = _FakeNet(0.7)
    monkeypatch.setattr(cv2.dnn, "readNet", lambda model_path: fake_net)

    def fake_decode(self: TextDetector, scores, geometry, conf_thresh):
        return [
            (5.0, 5.0, 25.0, 25.0),
            (600.0, 280.0, 630.0, 310.0),
        ], [0.5, 0.6]

    monkeypatch.setattr(TextDetector, "_decode", fake_decode)
    monkeypatch.setattr(
        cv2.dnn,
        "NMSBoxes",
        lambda boxes, confidences, score, thresh: np.array([[0], [1]]),
    )

    detector = TextDetector("dummy.pb")
    frame = np.zeros((320, 640, 3), dtype=np.uint8)

    has_text, confidence, total_ratio, largest_ratio, boxes = detector.detect(frame)

    assert has_text is False
    assert confidence == pytest.approx(0.7)
    assert total_ratio == 0.0
    assert largest_ratio == 0.0
    assert boxes == [(5, 5, 25, 25), (600, 280, 630, 310)]
