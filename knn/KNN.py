import numpy as np
from math import sqrt
from collections import Counter


def knn_classifier(k, X_train, y_train, x):
    assert 1 <= k <= X_train.shape[0], 'k error'
    assert X_train.shape[0] == y_train.shape[0], 'size error'
    distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train]
    nearest = np.argsort(distances)
    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)
    return votes.most_common(1)[0][0]


class KNNClassifier:
    def __init__(self, k):
        """初始化kNN分类器"""
        assert 1 <= k, 'k must be valid'
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集训练kNN分类器"""
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], "the feature number of X_predict must be equal to X_train"
        y_predict = [self._predict(x_predict) for x_predict in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据x，返回表示x的预测结果值"""
        assert x.shape[0] == self._X_train.shape[1], 'the feature number of x must be equal X_train'
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def __repr__(self):
        return f'kNN(k={self.k})'
