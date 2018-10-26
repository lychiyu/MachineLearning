import numpy as np

from ..metrics import r2_score


class LinearRegression:
    def __init__(self):
        self.interception_ = None  # 截距
        self.coefficient_ = None  # 系数
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """训练模型"""
        # 横向上多加一列
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        # linalg.inv 转置
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coefficient_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """模型准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return 'LinerRegression()'
