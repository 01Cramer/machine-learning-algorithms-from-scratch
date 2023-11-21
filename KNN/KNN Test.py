# ---------------------------------------------------------------------------------- #
# In this file, I am comparing the predictions of my KNN algorithm implementation with the scikit-learn implementation for both regression and classification.


from KNN import KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

# ---------------------------------------------------------------------------------- #
neigh_reg = KNeighborsRegressor(n_neighbors=2)
neigh_reg.fit(X, y)
print(neigh_reg.predict([[1.5]]))
# Output 0.33
# ----------------------------------------------------------------------------------- #
knn_regressor = KNN(k=2, problem="reg")
knn_regressor.fit(X, y)
print(knn_regressor.predict([[1.5]]))
# Output 0.33
# ----------------------------------------------------------------------------------- #
neigh_class = KNeighborsClassifier(n_neighbors=2)
neigh_class.fit(X, y)
print(neigh_class.predict([[2.5]]))
# Output 0
# ----------------------------------------------------------------------------------- #
knn_class = KNN(k=2, problem="class")
knn_class.fit(X, y)
print(knn_class.predict([[2.5]]))
# Output 0
