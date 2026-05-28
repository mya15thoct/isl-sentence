"""
DTW k-NN classifier on bilstm2 embedding sequences.

Each sequence is (T_i, 64) — variable length.
DTW aligns two sequences temporally before computing distance.

Usage:
  clf = DTWClassifier(k=1)
  clf.fit(X_train, y_train)
  pred = clf.predict(x_test)
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _dtw_numba(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Multivariate DTW distance, normalized by path length.
    s1: (T1, D), s2: (T2, D)
    """
    T1, T2 = s1.shape[0], s2.shape[0]
    INF = 1e18

    dp = np.full((T1, T2), INF)
    dp[0, 0] = 0.0
    for i in range(T1):
        for j in range(T2):
            # Euclidean distance between frames
            d = 0.0
            for k in range(s1.shape[1]):
                diff = s1[i, k] - s2[j, k]
                d += diff * diff
            d = d ** 0.5

            if i == 0 and j == 0:
                dp[i, j] = d
            elif i == 0:
                dp[i, j] = d + dp[i, j - 1]
            elif j == 0:
                dp[i, j] = d + dp[i - 1, j]
            else:
                best = dp[i - 1, j]
                if dp[i, j - 1] < best:
                    best = dp[i, j - 1]
                if dp[i - 1, j - 1] < best:
                    best = dp[i - 1, j - 1]
                dp[i, j] = d + best

    return dp[T1 - 1, T2 - 1] / (T1 + T2)


def dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    return float(_dtw_numba(
        s1.astype(np.float64),
        s2.astype(np.float64),
    ))


class DTWClassifier:
    """
    k-NN classifier using DTW distance on variable-length sequences.

    fit(X, y):
        X — list of (T_i, 64) arrays
        y — (N,) int labels

    predict(x):
        x — (T, 64) array → predicted class int

    predict_proba(x):
        Returns per-class vote proportions (N_classes,)
    """

    def __init__(self, k: int = 1, n_classes: int = None):
        self.k         = k
        self.n_classes = n_classes
        self.X_train   = []
        self.y_train   = None

    def fit(self, X: list, y: np.ndarray):
        self.X_train   = X
        self.y_train   = np.array(y)
        if self.n_classes is None:
            self.n_classes = int(self.y_train.max()) + 1

    def _distances(self, x: np.ndarray) -> np.ndarray:
        return np.array([dtw_distance(x, s) for s in self.X_train])

    def predict(self, x: np.ndarray) -> int:
        dists = self._distances(x)
        nn    = np.argsort(dists)[:self.k]
        votes = np.bincount(self.y_train[nn], minlength=self.n_classes)
        return int(np.argmax(votes))

    def predict_top3(self, x: np.ndarray) -> list[int]:
        dists = self._distances(x)
        # For each class, take the nearest neighbour distance
        class_min = np.full(self.n_classes, np.inf)
        for i, d in enumerate(dists):
            c = self.y_train[i]
            if d < class_min[c]:
                class_min[c] = d
        return list(np.argsort(class_min)[:3])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        dists = self._distances(x)
        nn    = np.argsort(dists)[:self.k]
        votes = np.bincount(self.y_train[nn], minlength=self.n_classes)
        return votes / votes.sum()
