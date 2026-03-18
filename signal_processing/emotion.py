"""Emotion Detection - Multi-backend with automatic fallback

Priority chain:
  1. DeepFace  – most accurate, auto-downloads weights on first run
  2. FER       – lightweight, no separate download needed
  3. Geometry heuristic – always available, rule-based from image gradients

All backends return the same dict: {emotion: confidence (0–1), ...}
"""

from __future__ import annotations

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']


class EmotionDetector:
    """Multi-backend emotion detector with automatic fallback.

    Tries DeepFace → FER → geometry heuristic in that order.
    Once a backend fails at runtime it permanently falls back to the next tier.
    """

    def __init__(self):
        self.emotion_labels = EMOTION_LABELS
        self._backend = self._init_backend()
        logger.info(f'EmotionDetector backend: {self._backend}')

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, face_img, landmarks=None):
        """Predict emotion probabilities from a face crop.

        Args:
            face_img:  numpy array (H, W, 3) in BGR colour space
            landmarks: optional list of 468 (x, y) tuples – used by heuristic

        Returns:
            dict: {emotion: float confidence}  (values sum to ~1)
        """
        if face_img is None or face_img.size == 0:
            return self._uniform()

        if self._backend == 'deepface':
            return self._predict_deepface(face_img)
        elif self._backend == 'fer':
            return self._predict_fer(face_img)
        else:
            return self._predict_heuristic(face_img, landmarks)

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_backend(self) -> str:
        try:
            from deepface import DeepFace  # noqa: F401
            self._deepface_module = DeepFace
            return 'deepface'
        except ImportError:
            pass

        try:
            from fer import FER
            self._fer_detector = FER(mtcnn=False)
            return 'fer'
        except ImportError:
            pass

        logger.warning(
            'EmotionDetector: Neither DeepFace nor FER installed. '
            'Using geometry heuristic (install deepface or fer for better accuracy).'
        )
        return 'heuristic'

    # ------------------------------------------------------------------
    # Per-backend prediction
    # ------------------------------------------------------------------

    def _predict_deepface(self, face_img) -> dict:
        try:
            result = self._deepface_module.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            raw = result.get('emotion', {})
            total = sum(raw.values()) + 1e-9
            out = {}
            for label in self.emotion_labels:
                # DeepFace uses 'happy' not 'happiness' etc.
                key_map = {'happiness': 'happy', 'sadness': 'sad', 'anger': 'angry',
                           'disgust': 'disgust', 'fear': 'fear',
                           'surprise': 'surprise', 'neutral': 'neutral'}
                df_key = key_map.get(label, label)
                out[label] = float(raw.get(df_key, 0.0) / total)
            return out
        except Exception as e:
            logger.debug(f'DeepFace inference error: {e}. Falling back.')
            self._backend = 'fer' if hasattr(self, '_fer_detector') else 'heuristic'
            return self.predict(face_img)

    def _predict_fer(self, face_img) -> dict:
        try:
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            detections = self._fer_detector.detect_emotions(rgb)
            if detections:
                raw = detections[0].get('emotions', {})
                total = sum(raw.values()) + 1e-9
                return {
                    label: float(raw.get(label, 0.0) / total)
                    for label in self.emotion_labels
                }
            return self._uniform()
        except Exception as e:
            logger.debug(f'FER inference error: {e}. Falling back.')
            self._backend = 'heuristic'
            return self._predict_heuristic(face_img)

    def _predict_heuristic(self, face_img, landmarks=None) -> dict:
        """Lightweight geometry-based heuristic using image statistics.

        Analyses:
          - Mouth region variance  → open mouth = happiness / surprise
          - Eye region brightness  → heavy lids = sadness / fatigue
          - Forehead gradient mag  → furrowed brows = anger / fear
          - Overall activity level → low = neutral
        """
        try:
            h, w = face_img.shape[:2]
            if h < 20 or w < 20:
                return self._uniform()

            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

            mouth_roi  = gray[int(h * 0.65):,         int(w * 0.20):int(w * 0.80)]
            eye_roi    = gray[int(h * 0.20):int(h * 0.50), :]
            brow_roi   = gray[int(h * 0.08):int(h * 0.28), :]

            mouth_var  = float(np.var(mouth_roi))  if mouth_roi.size  > 0 else 0.0
            eye_bright = float(np.mean(eye_roi))   if eye_roi.size    > 0 else 128.0
            mouth_mean = float(np.mean(mouth_roi)) if mouth_roi.size  > 0 else 128.0

            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            brow_grad    = float(np.mean(mag[int(h*0.08):int(h*0.28), :])) if h > 0 else 0.0
            overall_grad = float(np.mean(mag))

            scores = {e: 0.10 for e in self.emotion_labels}

            # Open mouth (high variance) → happy or surprise
            if mouth_var > 400:
                scores['happiness'] += 0.35
                scores['surprise']  += 0.15
            if mouth_var > 1200:
                scores['surprise']  += 0.30
                scores['happiness'] -= 0.10

            # Bright mouth interior (open, teeth visible) → happiness
            if mouth_mean > 160:
                scores['happiness'] += 0.15

            # Droopy eyes (low brightness) → sadness
            if eye_bright < 90:
                scores['sadness'] += 0.25

            # Strong brow gradient (furrowing) → anger / fear
            if brow_grad > 25:
                scores['anger'] += 0.25
                scores['fear']  += 0.10
            if brow_grad > 40:
                scores['anger'] += 0.15

            # Low overall activity → neutral
            if overall_grad < 12:
                scores['neutral'] += 0.45
            elif overall_grad < 20:
                scores['neutral'] += 0.20

            # Clip negatives and normalise
            total = sum(max(v, 0.0) for v in scores.values()) + 1e-9
            return {k: round(float(max(v, 0.0) / total), 4) for k, v in scores.items()}

        except Exception as e:
            logger.debug(f'Heuristic emotion error: {e}')
            return self._uniform()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _uniform(self) -> dict:
        p = round(1.0 / len(self.emotion_labels), 4)
        return {e: p for e in self.emotion_labels}
