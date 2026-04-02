from __future__ import annotations

import numpy as np


def top_k_search(query: np.ndarray, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = matrix @ query
    if len(scores) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    k = min(k, len(scores))
    indices = np.argpartition(-scores, k - 1)[:k]
    indices = indices[np.argsort(-scores[indices])]
    return indices.astype(np.int64), scores[indices].astype(np.float32)
