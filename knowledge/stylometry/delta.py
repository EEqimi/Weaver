# knowledge/stylometry/delta.py
"""Burrows's Delta 与余弦距离（spec §8.1）。基于 scipy/numpy，不重造轮子。"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cosine


def zscore_matrix(X: np.ndarray) -> np.ndarray:
    """逐特征 z-score（零方差列置 0，避免除零）。"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def burrows_delta(a: np.ndarray, b: np.ndarray) -> float:
    """两个 z-score 化向量之间的 Burrows's Delta = 平均绝对差。"""
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))


def author_centroids(X: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    """返回每个作者（标签）的 z-score 特征质心。"""
    Z = zscore_matrix(X)
    out: dict[str, np.ndarray] = {}
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        out[lab] = Z[idx].mean(axis=0)
    return out


def classify_by_delta(X: np.ndarray, labels: list[str],
                      test_idx: list[int]) -> tuple[list[str], list[str]]:
    """用 Delta 最近质心做作者归属（留出法）：返回 (预测, 真实)。

    训练集 = 除 test_idx 外的全部样本；测试样本分配给质心距离最近的作者。
    """
    all_idx = set(range(len(labels)))
    train_idx = sorted(all_idx - set(test_idx))
    Z = zscore_matrix(X)
    centroids: dict[str, np.ndarray] = {}
    for lab in set(labels):
        idx = [i for i in train_idx if labels[i] == lab]
        centroids[lab] = Z[idx].mean(axis=0)
    preds, truths = [], []
    for i in test_idx:
        zi = Z[i]
        best = min(centroids, key=lambda lab: burrows_delta(zi, centroids[lab]))
        preds.append(best)
        truths.append(labels[i])
    return preds, truths
