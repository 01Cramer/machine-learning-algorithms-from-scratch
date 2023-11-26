import numpy as np


def euclidean_distance(X_observation, X_test_observation):
    euclidean_distance = 0
    if len(X_observation) != len(X_test_observation):
        print("Can't calculate distance between vectors of different size")
        return None
    else:
        for i in range(len(X_observation)):
            if isinstance(X_observation[i], str) or isinstance(
                X_test_observation[i], str
            ):
                print("Can't calculate distance between not encoded qualitative variables")
                return None
            else:
                euclidean_distance += (X_observation[i] - X_test_observation[i]) ** 2
    euclidean_distance = np.sqrt(euclidean_distance)

    return euclidean_distance
# ---------------------------------------------------------------------------------- #
class KNN:
    def __init__(self, k=3, problem="reg"):
        self.k = k
        self.problem = problem

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        predictions = [self.__predict(observation) for observation in X_test]

        return predictions

    def __predict(self, observation):
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

