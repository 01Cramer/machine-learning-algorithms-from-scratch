import numpy as np


def euclidean_distance(X_observation, X_test_observation):
    euclidean_distance = 0
    if len(X_observation) != len(X_test_observation):
        raise ValueError("Can't calculate distance between vectors of diffrent size")
    else:
        for i in range(len(X_observation)):
            if isinstance(X_observation[i], str) or isinstance(
                X_test_observation[i], str
            ):
                raise ValueError("Can't calculate distance between string variables")
            else:
                euclidean_distance += (X_observation[i] - X_test_observation[i]) ** 2

    return np.sqrt(euclidean_distance)


class KNN:
    def __init__(self, k=3, distance="euclidean"):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        predictions = [self._predict(X_test) for observation in X_test]
        return predictions

    def _predict():  # To complete
        return 0
