import numpy as np
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import r2_score
import pickle

class LinearRegression:

    def __init__(self) -> None:
        self.args = self.parse_arguments()
        self.weight = None
        self.bias = None
        self.x = []
        self.y = []
        self.x_scaled = []
        self.y_scaled = []
        self.columns = []

    def fit(self):
        n_samples, _ = self.x.shape
        self.weight = 0
        self.bias = 0
        for i in range(self.args.iterations):
            y_pred = self.x_scaled * self.weight + self.bias
            dw = np.sum((self.x_scaled) * (y_pred - self.y))/n_samples

            db = np.sum(y_pred - self.y)/n_samples
            self.weight = self.weight - (self.args.learning_rate * dw)
            self.bias = self.bias - (self.args.learning_rate * db)
        self.unscale()

    def save(self):
        try:
            with open('model.pkl', 'wb') as f:
                pickle.dump((self.weight, self.bias), f)
        except ValueError:
            AssertionError('Error while saving model.txt')

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Linear Regression")
        parser.add_argument("-p", "--path", help="csv dataset to load",
                            type=str, default="data.csv")
        parser.add_argument("-i", "--iterations", help="number of iterations",
                            type=int, default=1000)
        parser.add_argument("-lr", "--learning-rate", help="learning rate",
                            type=float, default=0.01)
        return (parser.parse_args())

    @staticmethod
    def load_data(path):
        try:
            df = pd.read_csv(path)
            print('here')
            print(f'Loading dataset of dimensions {df.shape}')
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            return X, y, df.columns
        except Exception as error:
            sys.exit(error)

    @staticmethod
    def scaling(data):
        len_ = len(data)
        mean = np.sum(data) / len_
        var = np.sum((data-mean) ** 2)
        std = sqrt(var/len_-1)
        return ((data - mean)/std)

    def unscale(self):
        len_ = len(self.x)
        mean = np.sum(self.x) / len_
        var = np.sum((self.x-mean) ** 2)
        std = sqrt(var/len_-1)
        self.bias = self.bias - (mean * self.weight/std)
        self.weight = self.weight/std

    def prepare_data(self):
        self.x, self.y, self.columns = self.load_data(self.args.path)
        self.x_scaled = self.scaling(self.x).flatten()
        self.y_scaled = self.scaling(self.y)

    def predict(self, X):

        y_pred = X * self.weight + self.bias
        print(22899 * self.weight + self.bias)
        print(r2_score(self.y, y_pred))
        return (y_pred)

    def plot_results(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.x, self.y, color="red", s=30)
        plt.plot(self.x, self.predict(self.x), color='green', linewidth=2,
                 label='Prediction')
        plt.show()


if __name__ == '__main__':
    lr_model = LinearRegression()
    lr_model.prepare_data()
    lr_model.fit()
    lr_model.save()
    lr_model.predict(lr_model.x)
    lr_model.plot_results()
