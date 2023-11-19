import numpy as np
import math


class NaiveBayes:
    def __init__(self):
        self.alpha = 0.001  # Laplace smoothing

    def fit(self, X, y):
        self.X = X
        self.y = y

        hypothesis_probability = {}  # P(h)
        classes = set()

        for i in range(len(self.y)):
            classes.add(y[i])

        self.classes = classes

        for category in classes:
            hypothesis_probability[category] = self.y.count(category) / len(y)

        self.hypothesis_probability = hypothesis_probability

    def predict(self, X_test):
        predictions = [self._predict(observation) for observation in X_test]

        return predictions

    def _predict(self, X_test_observation):
        class_probability = {}
        for category in self.classes:
            naive_probabilities = []
            for i in range(len(X_test_observation)):
                naive = X_test_observation[i]
                naive_counter = 0
                for j in range(len(self.X)):
                    if naive in self.X[j] and self.y[j] == category:
                        naive_counter += 1
                naive_probab = naive_counter / len(self.y)
                if naive_probab == 0:
                    naive_probab = self.alpha
                naive_probabilities.append(naive_probab)
            class_probability[category] = (
                math.prod(naive_probabilities) * self.hypothesis_probability[category]
            )
        print(self.hypothesis_probability)
        print(class_probability)

print("hello")