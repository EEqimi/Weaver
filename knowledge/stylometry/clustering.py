# knowledge/stylometry/clustering.py
"""降维与聚类（spec §8.1）：PCA 与层次聚类。基于 scipy/scikit-learn。"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA


def pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """返回 (投影后坐标, 各主成分解释方差比)。"""
    model = PCA(n_components=n_components)
    return model.fit_transform(X), model.explained_variance_ratio_


def hierarchical_linkage(X: np.ndarray, method: str = "ward"):
    """返回 scipy 层次聚类 linkage 矩阵。"""
    return linkage(X, method=method)


def cluster_labels(X: np.ndarray, n_clusters: int, method: str = "ward") -> np.ndarray:
    """把样本切成 n_clusters 个簇，返回簇标签数组。"""
    Z = hierarchical_linkage(X, method=method)
    return fcluster(Z, t=n_clusters, criterion="maxclust")
