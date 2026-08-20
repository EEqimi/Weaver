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

from .extract import StylometricVectorizer


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

    注意：本函数接收**已向量化**的特征矩阵 X。若 X 的词汇表在分组 CV 之前
    用全部作品拟合，则左出作品会参与词汇选择，造成特征选择泄漏。对原始文本
    做泄漏安全的 CV 请使用 grouped_cross_validation_texts（每折重拟合向量器）。

    返回每折准确率列表。
    """
    y, _ = _labels_to_int(labels)
    groups = np.array([_work_group_id(w, work_ids) for w in work_ids])
    clf = _make_classifier(classifier)
    n_folds = cv if cv is not None else len(set(work_ids))
    return list(cross_val_score(clf, X, y, groups=groups,
                                cv=GroupKFold(n_splits=n_folds)))


def grouped_cross_validation_texts(texts: list[str], labels: list[str],
                                   work_ids: list[str], classifier: str = "svm",
                                   cv: int | None = None,
                                   vectorizer: StylometricVectorizer | None = None):
    """按作品分组的泄漏安全交叉验证（Phase 3–4.1 task item 2）。

    对每一折：
        训练作品 → fit StylometricVectorizer → transform 折内训练数据 → fit 分类器
        左出作品 → 用该折拟合的向量器 transform → 评估
    因此左出作品绝不参与字符/词 unigram 的词汇选择（杜绝特征选择泄漏）。

    返回每折准确率列表（GroupKFold 顺序）。
    """
    texts = list(texts)
    labels = list(labels)
    work_ids = list(work_ids)
    y, _ = _labels_to_int(labels)
    groups = np.array([_work_group_id(w, work_ids) for w in work_ids])
    n_folds = cv if cv is not None else len(set(work_ids))
    base = vectorizer if vectorizer is not None else StylometricVectorizer()
    accs: list[float] = []
    for train_idx, val_idx in GroupKFold(n_splits=n_folds).split(texts, y, groups):
        # 每折独立拟合向量器（使用与 base 相同的配置，避免共享已拟合的词汇表）
        fold_vec = StylometricVectorizer(
            char_n=base.char_n, char_top_k=base.char_top_k,
            word_top_k=base.word_top_k, function_words=base.function_words,
        )
        X_train = fold_vec.fit_transform([texts[i] for i in train_idx])
        X_val = fold_vec.transform([texts[i] for i in val_idx])
        clf = _make_classifier(classifier)
        clf.fit(X_train, y[train_idx])
        preds = clf.predict(X_val)
        accs.append(float((preds == y[val_idx]).mean()))
    return accs


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
