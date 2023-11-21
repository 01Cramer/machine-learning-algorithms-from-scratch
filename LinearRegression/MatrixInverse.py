import numpy as np


def _find_row_with_most_zeros(matrix):
    max_zeros = 0
    row = 0
    for i in range(matrix.shape[0]):
        zeros = 0
        for j in range(matrix.shape[1]):
            if matrix[i][j] == 0:
                zeros += 1
        if zeros > max_zeros:
            row = i
            max_zeros = zeros
    return row

def find_det(matrix):
    if matrix.shape[0] == 1:
        return matrix[0][0]
    elif matrix.shape[0] == 2:
        return matrix[0][0]*matrix[1][1] - matrix[1][0]*matrix[0][1]
    else:
        row_with_most_zeros = _find_row_with_most_zeros(matrix)
        det = 0
        for j in range(matrix.shape[1]):
            aij = matrix[row_with_most_zeros][j]
            if aij == 0:
                continue
            else:
                Aij = (-1)**(row_with_most_zeros+j)
                sub_matrix = matrix
                sub_matrix = np.delete(sub_matrix, row_with_most_zeros, axis=0)
                sub_matrix = np.delete(sub_matrix, j, axis=1)
                Aij *= find_det(sub_matrix)

                det += Aij * aij
        return det

def find_complementary(matrix):
    complementary_matrix = np.array([])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            sub_matrix = matrix
            sub_matrix = np.delete(sub_matrix, row, axis=0)
            sub_matrix = np.delete(sub_matrix, col, axis=1)

            value = find_det(sub_matrix)
            value *= (-1) ** (row+col)
            complementary_matrix = np.append(complementary_matrix, value)

    complementary_matrix = complementary_matrix.reshape((matrix.shape[0], matrix.shape[1]))
    return complementary_matrix

def transpose(matrix):
    sub_matrix = np.array([])
    for col in range(matrix.shape[1]):
        for row in range(matrix.shape[0]):
            sub_matrix = np.append(sub_matrix, matrix[row][col])
    sub_matrix = sub_matrix.reshape((matrix.shape[1], matrix.shape[0]))
    return sub_matrix

def inv_laplace_method(matrix):
    det = find_det(matrix)
    transposed_complementary = transpose(find_complementary(matrix))
    inversed = transposed_complementary / det
    return inversed

matrix = np.array([[2,5,7], [6,3,4], [5,-2,-3]])
inversed = inv_laplace_method(matrix)

print(matrix)
print(inversed)