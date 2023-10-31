import numpy as np


class NaiveBayes:
    def __init__(self):
        self.alpha = 0.001  # Laplace smoothing

    def fit(self, X, y):
        self.X = X
        self.y = y

        hypothesis_probability = {}  # P(h)
        categories = set()

        for i in range(len(self.y)):
            categories.add(y[i])

        self.categories = categories

        for category in categories:
            hypothesis_probability[category] = self.y.count(category) / len(y)

        self.hypothesis_probability = hypothesis_probability

    def predict(self, X_test):
        predictions = [self._predict(observation) for observation in X_test]

        return predictions

    def _predict(self, X_test_observation):
        class_probability = {}
        for category in self.categories:
            naive_probabilites = []  # P(D|h)

            rows_of_category = []
            for i in range(len(self.y)):
                if self.y[i] == category:
                    rows_of_category.append(self.X[i])

            print(rows_of_category)  # Debuging
            for list in rows_of_category:
                print(list.count("żywe"))
