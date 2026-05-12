"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV


class HallucinationProbe:
    """Binary classifier that detects hallucinations from hidden-state features."""

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()
        self._pca = None
        self.log_reg = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features with ``StandardScaler``.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        X_scaled = self._scaler.fit_transform(X)
        
        n_components = min(
            128,
            X_scaled.shape[0] - 1,
            X_scaled.shape[1],
        )

        self._pca = PCA(n_components=n_components, random_state=42)
        X_pca = self._pca.fit_transform(X_scaled)

        X_t = torch.from_numpy(X_pca).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        self.log_reg = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', max_iter=2000)
        self.log_reg.fit(X_t, y_t)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise accuracy.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        # Candidate thresholds: unique predicted probabilities plus a coarse grid.
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_accuracy = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = accuracy_score(y_val, y_pred_t)
            if score > best_accuracy:
                best_accuracy = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        if self._pca is None:
            raise RuntimeError("PCA has not been fitted yet. Call fit() first.")

        X_scaled = self._scaler.transform(X)
        X_pca = self._pca.transform(X_scaled)

        return self.log_reg.predict_proba(X_pca)