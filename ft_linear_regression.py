import numpy as np


class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iters):
            y_pred = X.dot(self.weight) + self.bias
            dw = ((X.T).dot(y_pred - y))/n_samples
            db = sum(y_pred - y)/n_samples

            self.weight = self.weight - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)

    def predict(self, X):
        y_pred = X.dot(self.weight) + self.bias

        return (y_pred)
