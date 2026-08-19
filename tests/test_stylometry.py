# tests/test_stylometry.py
"""Layer D 文体学测试（spec §12）：提取、Burrows Delta、PCA、切分与泄漏防护。"""
import numpy as np
import pytest

from knowledge.stylometry.clustering import pca
from knowledge.stylometry.delta import burrows_delta, cosine_distance, zscore_matrix
from knowledge.stylometry.extract import FUNCTION_WORDS, StylometricVectorizer
from knowledge.stylometry.validation import (
    evaluate_heldout, grouped_cross_validation, split_by_work,
)


# ---- 提取 ----
def test_stylometric_vectorizer_shape():
    texts = ["the cat sat on the mat", "he walked alone down the lane",
             "she went home quickly", "they came and went"]
    X = StylometricVectorizer().fit_transform(texts)
    assert X.shape[0] == len(texts)
    assert X.shape[1] >= len(FUNCTION_WORDS)
    assert np.all(np.isfinite(X))


def test_stylometric_vectorizer_requires_fit():
    with pytest.raises(RuntimeError):
        StylometricVectorizer().transform(["x"])


# ---- Burrows Delta ----
def test_burrows_delta_identical_is_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert burrows_delta(a, a) == 0.0


def test_burrows_delta_distinct_is_positive():
    assert burrows_delta(np.array([0.0, 0.0]), np.array([1.0, 1.0])) > 0.0


def test_cosine_distance_identical_is_zero():
    assert cosine_distance(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.0, abs=1e-6)


def test_zscore_matrix_zero_variance_column():
    X = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    Z = zscore_matrix(X)
    assert Z.shape == X.shape
    assert np.allclose(Z[:, 0], 0.0)  # 常数列 → 0


# ---- PCA ----
def test_pca_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 10))
    proj, ratio = pca(X, n_components=2)
    assert proj.shape == (20, 2)
    assert ratio.shape == (2,)
    assert np.all(ratio >= 0) and ratio.sum() <= 1.0 + 1e-6


# ---- 切分与泄漏防护 ----
def test_split_by_work_separates_held_out():
    works = ["a1", "a2", "d1", "d2", "d2"]
    train, test = split_by_work(works, ["d2"])
    assert set(works[i] for i in test) == {"d2"}
    assert all(works[i] != "d2" for i in train)


def test_held_out_never_leaks_into_train():
    # 作者标签按 work 分组；held-out 作品的 chunk 绝不能进入训练
    labels = ["austen"] * 20 + ["dickens"] * 20
    works = (["pride_and_prejudice"] * 10 + ["emma"] * 10
             + ["great_expectations"] * 10 + ["tale_of_two_cities"] * 10)
    train, test = split_by_work(works, ["tale_of_two_cities"])
    for i in train:
        assert works[i] != "tale_of_two_cities"
    assert all(works[i] == "tale_of_two_cities" for i in test)


def _separable(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 10)) * 0.05
    X[:20, 0] += 1.0   # austen 特征 0 高
    X[20:, 1] += 1.0   # dickens 特征 1 高
    labels = ["austen"] * 20 + ["dickens"] * 20
    works = (["pride_and_prejudice"] * 10 + ["emma"] * 10
             + ["great_expectations"] * 10 + ["david_copperfield"] * 10)
    return X, labels, works


def test_grouped_cross_validation_returns_scores():
    X, labels, works = _separable()
    scores = grouped_cross_validation(X, labels, works, classifier="svm")
    assert len(scores) == 4
    assert all(s >= 0.9 for s in scores)  # 数据可分 → 高准确率


def test_evaluate_heldout():
    X, labels, works = _separable()
    acc, preds, truths = evaluate_heldout(
        X, labels, works, ["david_copperfield"], classifier="svm")
    assert acc == 1.0
    assert set(preds) == {"dickens"}
