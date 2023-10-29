from KNN import KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

# ----------------------------------------------------------------------------------#
neigh_reg = KNeighborsRegressor(n_neighbors=3)
neigh_reg.fit(X, y)
print(neigh_reg.predict([[1.5]]))
# Output 0.33
# -----------------------------------------------------------------------------------#
knn_regressor = KNN()
knn_regressor.fit(X, y)
print(knn_regressor.predict([[1.5]]))
# Output 0.33
# -----------------------------------------------------------------------------------#
neigh_class = KNeighborsClassifier(n_neighbors=3)
neigh_class.fit(X, y)
print(neigh_class.predict([[1.1]]))
# Output 0
# -----------------------------------------------------------------------------------#
knn_class = KNN(problem="class")
knn_class.fit(X, y)
print(knn_class.predict([[1.1]]))
# Output 0
