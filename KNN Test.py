from KNN import KNN
from sklearn.neighbors import KNeighborsRegressor

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

# ----------------------------------------------------------------------------------#
neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(X, y)
print(neigh.predict([[1.5]]))
# Output 0.33
# -----------------------------------------------------------------------------------#
knn_regressor = KNN()
knn_regressor.fit(X, y)
print(knn_regressor.predict([[1.5]]))
# Output 0.33
