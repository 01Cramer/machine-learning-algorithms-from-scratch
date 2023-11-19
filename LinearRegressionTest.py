from LinearRegressionWithNormalEquation import LinearRegressionNormalEquation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, 1:2].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1)

model = LinearRegressionNormalEquation()
model.fit(X_train, y_train)

sklearn = LinearRegression()
sklearn.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = model.mse(y_test)

sklearn_predictions = sklearn.predict(X_test)

sklearn_mse = mean_squared_error(y_test, sklearn_predictions)

print("OUR MSE:")
print(mse)

print("SKLEARN MSE:")
print(sklearn_mse)

plt.plot(X_test, predictions.matrix, "r-", label="Pred")
plt.plot(X_train, y_train, "b.")
plt.show()