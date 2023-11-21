from MatrixOperations import Matrix
from sklearn.preprocessing import add_dummy_feature

class LinearRegressionNormalEquation:
    def __init__(self):
        self.theta = None
        self.X = None
        self.y = None
        self.predictions = None

    def __str__(self):
        return str(self.theta)

    def fit(self, X, y):
        X = add_dummy_feature(X)
        self.X = Matrix(X)
        self.y = Matrix(y)
        self._calculate_theta()

    def _calculate_theta(self):
        X_tran = Matrix(self.X.matrix)
        X_tran.tran()
        self.theta = X_tran.mul(self.X).inv().mul(X_tran).mul(self.y)

    def predict(self, X_input):
        X_input = add_dummy_feature(X_input)    # need to include intercept (beta 0) in X matrix
        X_input = Matrix(X_input)

        y_predict = X_input.mul(self.theta)
        self.predictions = y_predict
        return y_predict

    def mse(self, y_true):
        n = len(self.predictions.matrix)
        rss = 0
        for i in range(n):
            rss += (y_true[i] - self.predictions.matrix[i]) ** 2

        mse = rss / n
        return mse

