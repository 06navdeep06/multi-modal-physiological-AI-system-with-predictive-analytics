"""Face Detection using MediaPipe FaceDetection."""

from __future__ import annotations

import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.7):
        self._mp = mp.solutions.face_detection
        self._detector = self._mp.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, frame) -> list[tuple[int, int, int, int]]:
        """Detect faces and return bounding boxes as (x, y, w, h) tuples."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        bboxes = []
        if results.detections:
            ih, iw = frame.shape[:2]
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                x = int(bb.xmin * iw)
                y = int(bb.ymin * ih)
                w = int(bb.width  * iw)
                h = int(bb.height * ih)
                # Clamp to frame boundaries
                x = max(0, x);  y = max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                if w > 0 and h > 0:
                    bboxes.append((x, y, w, h))
        return bboxes
