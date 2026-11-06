from __future__ import annotations

import numpy as np


class OneClass:
    """Mahalanobis distance to the healthy records (ridge-regularised covariance)."""

    def __init__(self, ridge: float = 1.0):
        self.ridge = ridge

    def fit(self, F: np.ndarray):
        self.mu = F.mean(0)
        self.sd = F.std(0) + 1e-9
        Z = (F - self.mu) / self.sd
        C = np.cov(Z.T) + self.ridge * np.eye(Z.shape[1])
        self.P = np.linalg.inv(C)
        return self

    def score(self, F: np.ndarray) -> np.ndarray:
        Z = (F - self.mu) / self.sd
        return np.einsum("ij,jk,ik->i", Z, self.P, Z)


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    s = np.sort(cal_scores)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= n else float("inf")
