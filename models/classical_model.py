"""Classical ML Stress Predictor

Wraps scikit-learn classifiers with:
  - Graceful load / no-model-file handling
  - Rule-based heuristic fallback so the system is always functional
  - train() convenience method that persists the fitted model with joblib
"""

from __future__ import annotations

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ClassicalModel:
    """Logistic Regression / Random Forest wrapper with rule-based fallback.

    When no trained model file exists, `predict()` delegates to a deterministic
    heuristic that combines physiological signals into a stress score.

    Args:
        model_type: 'logreg' | 'rf'
        model_path: Path to a joblib-serialised fitted model (optional)
    """

    def __init__(self, model_type: str = 'logreg', model_path: str = None):
        self.model_type = model_type
        self.model_path = model_path
        self._model = None
        self._trained = False

        self._model = self._build_model(model_type)
        if model_path and os.path.isfile(model_path):
            self._load(model_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, X):
        """Return class probability array of shape (n_samples, n_classes).

        Falls back to rule-based heuristic if no model is trained.

        Args:
            X: numpy array (n_samples, n_features)

        Returns:
            numpy array (n_samples, 3): [p_low, p_medium, p_high]
        """
        if self._trained:
            try:
                if hasattr(self._model, 'predict_proba'):
                    return self._model.predict_proba(X)
                preds = self._model.predict(X)
                # Convert class indices to one-hot probabilities
                n = len(preds)
                proba = np.zeros((n, 3))
                for i, p in enumerate(preds):
                    proba[i, int(p)] = 1.0
                return proba
            except Exception as e:
                logger.warning(f'Model predict error: {e}. Using heuristic.')

        return self._heuristic_predict(X)

    def train(self, X, y):
        """Fit the model and save to disk.

        Args:
            X: numpy array (n_samples, n_features)
            y: array-like of integer labels {0=Low, 1=Medium, 2=High}
        """
        try:
            import joblib
            self._model.fit(X, y)
            self._trained = True
            if self.model_path:
                os.makedirs(os.path.dirname(self.model_path) or '.', exist_ok=True)
                joblib.dump(self._model, self.model_path)
                logger.info(f'Model saved to {self.model_path}')
        except Exception as e:
            logger.error(f'Model training error: {e}')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model(model_type: str):
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            if model_type == 'rf':
                return RandomForestClassifier(n_estimators=100, random_state=42,
                                              class_weight='balanced')
            return LogisticRegression(max_iter=500, random_state=42,
                                      class_weight='balanced')
        except ImportError:
            return None

    def _load(self, path: str):
        try:
            import joblib
            self._model = joblib.load(path)
            self._trained = True
            logger.info(f'Classical model loaded from {path}')
        except Exception as e:
            logger.warning(f'Could not load model from {path}: {e}')

    @staticmethod
    def _heuristic_predict(X) -> np.ndarray:
        """Rule-based stress scoring from fused feature vector.

        Maps physiological signals to a 0–1 stress score using domain knowledge:
          - High HR (> 90 BPM normalised) → stress increases
          - Low HRV RMSSD (< 30 ms normalised) → stress increases
          - High fatigue (> 0.6) → stress increases
          - High resp rate (> 18 BPM normalised) → slight stress increase

        Feature vector layout (mirrors FeatureFusion output):
          [hr_stats(6), hrv_stats(6), resp_stats(6), blink_stats(6), fatigue_stats(6)]
        Each channel block: [mean_z, cv, min_z, max_z, med_z, slope]
        We use index 0 (mean_z) of each channel.
        """
        n = X.shape[0]
        proba = np.zeros((n, 3))

        for i in range(n):
            row = X[i]
            # Pull channel means (index 0 of each 6-element block)
            hr_z     = float(row[0])  if len(row) > 0  else 0.0
            hrv_z    = float(row[6])  if len(row) > 6  else 0.0
            resp_z   = float(row[12]) if len(row) > 12 else 0.0
            fatigue_z = float(row[24]) if len(row) > 24 else 0.0

            # Stress proxy: higher HR + lower HRV + higher fatigue = more stress
            # Z-scores: positive = above mean, negative = below mean
            stress_score = (
                0.35 * np.clip( hr_z,      -2, 2) / 2.0 +   # high HR → stress
                0.35 * np.clip(-hrv_z,     -2, 2) / 2.0 +   # low HRV → stress
                0.20 * np.clip( fatigue_z, -2, 2) / 2.0 +   # high fatigue → stress
                0.10 * np.clip( resp_z,    -2, 2) / 2.0      # high RR → slight stress
            )
            # Map [-1, 1] → [0, 1]
            score = float(np.clip((stress_score + 1.0) / 2.0, 0.0, 1.0))

            if score < 0.35:
                proba[i] = [0.75, 0.20, 0.05]
            elif score < 0.65:
                proba[i] = [0.15, 0.70, 0.15]
            else:
                proba[i] = [0.05, 0.20, 0.75]

        return proba
