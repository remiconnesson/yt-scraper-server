"""Lightweight EAST text detection wrapper for slide frames."""

from __future__ import annotations

import cv2
import numpy as np

from slides_extractor.settings import EAST_MODEL_PATH, TEXT_CONF_THRESHOLD, TEXT_INPUT_SIZE

# Heuristics for slide-like text, not just small logos or background books.
MIN_TOTAL_AREA_RATIO = 0.015   # 1.5% of the frame covered by text
MIN_LARGEST_BOX_RATIO = 0.008  # Largest box covers 0.8% of the frame


class TextDetector:
    """Perform high-recall text detection using the EAST model."""

    def __init__(self, model_path: str = EAST_MODEL_PATH) -> None:
        self.net = cv2.dnn.readNet(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.layer_names = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3",
        ]

    def _decode(
        self, scores: np.ndarray, geometry: np.ndarray, conf_thresh: float
    ) -> tuple[list[tuple[float, float, float, float]], list[float]]:
        detections: list[tuple[float, float, float, float]] = []
        confidences: list[float] = []

        height, width = scores.shape[2], scores.shape[3]

        for y in range(height):
            scores_row = scores[0, 0, y]
            x0 = geometry[0, 0, y]
            x1 = geometry[0, 1, y]
            x2 = geometry[0, 2, y]
            x3 = geometry[0, 3, y]
            angles = geometry[0, 4, y]

            for x in range(width):
                score = scores_row[x]
                if score < conf_thresh:
                    continue

                angle = angles[x]
                cos_a = float(np.cos(angle))
                sin_a = float(np.sin(angle))

                height_box = x0[x] + x2[x]
                width_box = x1[x] + x3[x]

                offset_x = x * 4.0
                offset_y = y * 4.0

                center_x = offset_x + cos_a * x1[x] + sin_a * x2[x]
                center_y = offset_y - sin_a * x1[x] + cos_a * x2[x]

                x1_box = center_x - width_box / 2
                y1_box = center_y - height_box / 2
                x2_box = center_x + width_box / 2
                y2_box = center_y + height_box / 2

                detections.append((x1_box, y1_box, x2_box, y2_box))
                confidences.append(float(score))

        return detections, confidences

    def detect(self, frame: np.ndarray) -> tuple[bool, float, float, float]:
        """Detect text presence in a frame.

        Args:
            frame: Input image in BGR or RGB format.

        Returns:
            A tuple containing:
                - Whether slide-like text is present.
                - Maximum detection confidence from EAST.
                - Total text area ratio across all boxes.
                - Largest single box area ratio.
        """

        orig_h, orig_w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            1.0,
            TEXT_INPUT_SIZE,
            (123.68, 116.78, 103.94),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        scores, geometry = self.net.forward(self.layer_names)

        max_confidence = float(np.max(scores))

        boxes, confidences = self._decode(scores, geometry, TEXT_CONF_THRESHOLD)
        if not boxes:
            return False, max_confidence, 0.0, 0.0

        rects = [
            (x1, y1, x2 - x1, y2 - y1)
            for x1, y1, x2, y2 in boxes
        ]
        indices = cv2.dnn.NMSBoxes(rects, confidences, TEXT_CONF_THRESHOLD, 0.55)
        if len(indices) == 0:
            return False, max_confidence, 0.0, 0.0

        scale_x = orig_w / float(TEXT_INPUT_SIZE[0])
        scale_y = orig_h / float(TEXT_INPUT_SIZE[1])

        total_area = 0.0
        largest_area = 0.0

        for idx in indices.flatten():
            x1, y1, x2, y2 = boxes[idx]

            x1 = max(0, int(x1 * scale_x))
            y1 = max(0, int(y1 * scale_y))
            x2 = min(orig_w, int(x2 * scale_x))
            y2 = min(orig_h, int(y2 * scale_y))

            if x2 <= x1 or y2 <= y1:
                continue

            area = float((x2 - x1) * (y2 - y1))
            total_area += area
            largest_area = max(largest_area, area)

        if total_area == 0.0:
            return False, max_confidence, 0.0, 0.0

        frame_area = float(orig_w * orig_h)
        total_ratio = total_area / frame_area
        largest_ratio = largest_area / frame_area

        has_slide_text = (
            total_ratio >= MIN_TOTAL_AREA_RATIO
            or largest_ratio >= MIN_LARGEST_BOX_RATIO
        )

        return has_slide_text, max_confidence, total_ratio, largest_ratio
