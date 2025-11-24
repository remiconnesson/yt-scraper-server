"""Lightweight EAST text detection wrapper for slide frames."""

from __future__ import annotations

import cv2
import numpy as np

from slides_extractor.settings import EAST_MODEL_PATH, TEXT_CONF_THRESHOLD, TEXT_INPUT_SIZE


class TextDetector:
    """Perform high-recall text detection using the EAST model."""

    def __init__(self, model_path: str = EAST_MODEL_PATH) -> None:
        self.net = cv2.dnn.readNet(model_path)
        self.layer_names = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3",
        ]

    def detect(self, frame: np.ndarray) -> tuple[bool, float]:
        """Detect text presence in a frame.

        Args:
            frame: Input image in BGR or RGB format.

        Returns:
            A tuple indicating whether text is present and the maximum
            confidence observed.
        """

        blob = cv2.dnn.blobFromImage(
            frame,
            1.0,
            TEXT_INPUT_SIZE,
            (123.68, 116.78, 103.94),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        scores, _geometry = self.net.forward(self.layer_names)

        max_confidence = float(np.max(scores))
        has_text = max_confidence > TEXT_CONF_THRESHOLD

        return has_text, max_confidence
