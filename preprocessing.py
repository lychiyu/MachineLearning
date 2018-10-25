import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """根据训练集X获得均值和方差"""
        assert X.ndim == 2, 'the dimension of X must be 2'
        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])
        return self

    def transform(self, X):
        """进行均值归一化处理"""
        assert X.ndim == 2, 'the dimension of X must be 2'
        res_X = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            res_X[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return res_X
