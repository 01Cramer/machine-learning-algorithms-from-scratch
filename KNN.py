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
    def __init__(self, k=3, problem="reg"):
        self.k = k
        self.problem = problem

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        predictions = [self._predict(observation) for observation in X_test]

        return predictions

    def find_nearest(self, vector):
        min_val = vector[0]
        min_index = 0
        for i in range(1, len(vector)):
            if vector[i] < min_val:
                min_val = vector[i]
                min_index = i

        return min_index

    def _predict(self, observation):
        distances = {}
        k_nearest = []

        for i in range(len(self.X)):
            distances[i] = euclidean_distance(self.X[i], observation)

        sorted_distances = sorted(distances.items(), key=lambda x: x[1])

        for i in range(self.k):
            k_nearest.append(self.y[sorted_distances[i][0]])

        if self.problem == "reg":
            return np.mean(k_nearest)

        else:
            return max(set(k_nearest), key=k_nearest.count)
