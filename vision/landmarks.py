"""Facial Landmark Detection using MediaPipe FaceMesh (468 points)."""

from __future__ import annotations

import cv2
import mediapipe as mp


class FaceLandmarks:
    def __init__(self, static_mode: bool = False, max_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self._mp = mp.solutions.face_mesh
        self._mesh = self._mp.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def get_landmarks(self, frame) -> list[list[tuple[int, int]]]:
        """Return pixel-space (x, y) landmark lists for each detected face."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        all_faces = []
        if results.multi_face_landmarks:
            ih, iw = frame.shape[:2]
            for face in results.multi_face_landmarks:
                pts = [
                    (int(lm.x * iw), int(lm.y * ih))
                    for lm in face.landmark
                ]
                all_faces.append(pts)
        return all_faces
