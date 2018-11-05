import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components  # n 个主成分
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        """获取数据集X的前n的主成分"""

        def demean(X):
            """均值归0"""
            return X - np.mean(X, axis=0)

        def f(w, X):
            """目标函数"""
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            """目标函数的梯度"""
            return X.T.dot(X.dot(w)) * 2 / len(X)

        def direction(w):
            """w方向向量"""
            return w / np.linalg.norm(w)

        def gradient_ascent(df, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
            """梯度上升"""
            w = direction(initial_w)
            cur_iter = 0
            while cur_iter < n_iters:
                gradient_ascent = df(w, X)
                last_w = w
                w = w + eta * gradient_ascent
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break
                cur_iter += 1
            return w

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = gradient_ascent(df, X_pca, initial_w, eta, n_iters)
            self.components_[i, :] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
        return self

    def transform(self, X):
        """将给定的X，映射到各个主成分分量中: 高维数据映射到低维数据"""
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的X，反向映射回原来的特征空间: 低维数据映射到高维数据"""
        return X.dot(self.components_)
