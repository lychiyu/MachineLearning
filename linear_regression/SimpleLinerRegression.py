import numpy as np


class SimpleLinerRegression1:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        # 初始化求a中的分子分母
        numerator = 0.0
        denominator = 0.0
        # 分子分母求和
        for x_i, y_i in zip(x_train, y_train):
            numerator += (x_i - x_mean) * (y_i - y_mean)
            denominator += (x_i - x_mean) ** 2
        self.a_ = numerator / denominator
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_sigle):
        return self.a_ * x_sigle + self.b_


class SimpleLinerRegression2:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        # 初始化求a中的分子分母
        # 分子分母向量化运算
        numerator = np.array(x_train - x_mean).dot(y_train - y_mean)
        denominator = np.array(x_train - x_mean).dot(x_train - x_mean)
        self.a_ = numerator / denominator
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_sigle):
        return self.a_ * x_sigle + self.b_
