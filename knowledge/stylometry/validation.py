# knowledge/stylometry/validation.py
"""监督验证（spec §8.1 / §19.4）：SVM、逻辑回归、交叉验证与泄漏防护。

关键科学护栏：held-out work 绝不能进入训练（防止"同一作品内相邻 chunk"的
泄漏使作者识别虚高）。默认按"留出作品"切分，而非随机切分。
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.svm import SVC


def split_by_work(work_ids: list[str], held_out_works: list[str]) -> tuple[list[int], list[int]]:
    """按作品切分：held_out_works 全部作为测试，其余作为训练（无作品泄漏）。"""
    test_idx = [i for i, w in enumerate(work_ids) if w in set(held_out_works)]
    train_idx = [i for i, w in enumerate(work_ids) if w not in set(held_out_works)]
    if not test_idx or not train_idx:
        raise ValueError("held_out_works 切分后训练/测试集为空")
    return train_idx, test_idx


def _labels_to_int(labels: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    uniq = sorted(set(labels))
    mapping = {u: i for i, u in enumerate(uniq)}
    return np.array([mapping[l] for l in labels]), mapping


def evaluate_heldout(X: np.ndarray, labels: list[str], work_ids: list[str],
                     held_out_works: list[str], classifier: str = "svm"):
    """留出作品验证：训练于非 held-out 作品，测试于 held-out 作品。

    返回 (accuracy, 预测标签, 真实标签)。classifier ∈ {"svm", "logreg"}。
    """
    train_idx, test_idx = split_by_work(work_ids, held_out_works)
    y, mapping = _labels_to_int(labels)
    clf = _make_classifier(classifier)
    clf.fit(X[train_idx], y[train_idx])
    preds_int = clf.predict(X[test_idx])
    acc = float((preds_int == y[test_idx]).mean())
    inv = {v: k for k, v in mapping.items()}
    return acc, [inv[p] for p in preds_int], [labels[i] for i in test_idx]


def grouped_cross_validation(X: np.ndarray, labels: list[str], work_ids: list[str],
                             classifier: str = "svm", cv: int | None = None):
    """按作品分组的交叉验证（GroupKFold，杜绝同一作品 chunk 跨 fold 泄漏）。

    返回每折准确率列表。
    """
    y, _ = _labels_to_int(labels)
    groups = np.array([_work_group_id(w, work_ids) for w in work_ids])
    clf = _make_classifier(classifier)
    n_folds = cv if cv is not None else len(set(work_ids))
    return list(cross_val_score(clf, X, y, groups=groups,
                                cv=GroupKFold(n_splits=n_folds)))


def _make_classifier(kind: str):
    # class_weight="balanced"：作者 chunk 数不平衡时（如 Austen 833 vs Dickens 1495）
    # 若不均衡，线性分类器会坍缩到多数类（分组 CV 出现 0.0 准确率）。这是
    # 分类器自身对类别先验的处理，与任何 held-out 调参无关，train/heldout 一致。
    if kind == "svm":
        return SVC(kernel="linear", class_weight="balanced")
    if kind == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    raise ValueError(f"未知分类器: {kind}")


def _work_group_id(work: str, work_ids: list[str]) -> int:
    uniq = sorted(set(work_ids))
    return uniq.index(work)
