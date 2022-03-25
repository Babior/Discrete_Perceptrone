import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=100, random_state=1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rand = np.random.RandomState(self.random_state)
        self.weights = rand.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for x, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(x))
                self.weights[1:] += update * x
                self.weights[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)

        return self

    def net_input(self, X):
        z = np.dot(X, self.weights[1:]) + self.weights[0]
        return z

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)

    # Make a prediction with weights
    def get_single_prediction(self, row, weights):
        activation = weights[0]
        for i in range(len(row) - 1):
            activation += weights[i + 1] * row[i]
        return 1.0 if activation >= 0.0 else 0.0
