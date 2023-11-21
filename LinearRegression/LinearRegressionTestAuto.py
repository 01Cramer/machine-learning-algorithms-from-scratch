from LinearRegressionWithNormalEquation import LinearRegressionNormalEquation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Auto.csv')

X = data.iloc[:, [5, 4]].values
y = data.iloc[:, [3]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1)

model = LinearRegressionNormalEquation()
model.fit(X_train, y_train)

sklearn = LinearRegression()
sklearn.fit(X_train, y_train)

print(X_test)
predictions = model.predict(X_test)
print(predictions)
mse = model.mse(y_test)

sklearn_predictions = sklearn.predict(X_test)

sklearn_mse = mean_squared_error(y_test, sklearn_predictions)

print("OUR MSE:")
print(mse)

print("SKLEARN MSE:")
print(sklearn_mse)

print(model.theta)

X = np.array([[9], [45]])
Y = np.array([[1500], [5000]])
XY = np.append(X, Y, axis = 1)
RE, IM = np.meshgrid(X_test[:, [0]], X_test[:, [1]]) #2D planes
RE, IM = np.meshgrid(X, Y)
predictXY = model.predict(XY)
fig = plt.figure(figsize=(16, 12))
ax = plt.subplot(projection='3d')
ax.scatter(X_test[:, [0]], X_test[:, [1]], y_test, edgecolors = 'r')
srf = ax.plot_surface(np.array([[9.6,29],[5,15]]), np.array([[5000,3704],[4222,1500]]), np.array([[200,56]]), cstride=1, rstride=1)
ax.set_xlabel("mpg")
ax.set_ylabel("weight")
ax.set_zlabel("horsepower")
ax.set_title('3D PLOT')
ax.view_init(30, 50)
plt.show()