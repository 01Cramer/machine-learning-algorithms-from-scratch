import numpy as np
class Matrix:
    def __init__(self, array):
        self.matrix = np.array(array, dtype=float)
        self.cols = self.matrix.shape[1]
        self.rows = self.matrix.shape[0]

    def __str__(self):
        return str(self.matrix)

    def mul(self, multiplier):
        if isinstance(multiplier, (int, float)):  # multiplier is scalar
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] *= multiplier

            return Matrix(self.matrix)

        elif self.cols == multiplier.rows:  # multiplier is matrix
            m_output = np.array([])

            for m in range(self.rows):
                output_row = np.array([])
                for i in range(multiplier.cols):
                    value = 0
                    for j in range(self.cols):
                        value += self.matrix[m][j] * multiplier.matrix[j][i]
                    output_row = np.append(output_row, value)
                m_output = np.append(m_output, output_row)

            return Matrix(m_output.reshape(self.rows, multiplier.cols))

        else:
            print("Wrong size")

    def add(self, to_add):
        if self.rows == to_add.rows and self.cols == to_add.cols:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] += to_add.matrix[i][j]

            return Matrix(self.matrix)

        else:
            print("Wrong size")

    def tran(self):
        tran_matrix = np.array([])

        for i in range(self.cols):
            tran_row = np.array([])
            for j in range(self.rows):
                tran_row = np.append(tran_row, self.matrix[j][i])
            tran_matrix = np.append(tran_matrix, tran_row)

        tran_matrix = tran_matrix.reshape(self.cols, self.rows)
        self.matrix = tran_matrix
        self.rows, self.cols = self.cols, self.rows


    def _add_identity_matrix(self):
        matrix_result = np.array([])
        for i in range(self.rows):
            sub_row = np.zeros(self.cols)
            sub_row[i] = 1
            sub_matrix = np.append(self.matrix[i], sub_row)
            matrix_result = np.append(matrix_result, sub_matrix)
        matrix_result = matrix_result.reshape(self.rows, 2 * self.cols)

        return Matrix(matrix_result)

    def inv(self):
        triangle_matrix = self._add_identity_matrix()
        for i in range(0, self.cols - 1):
            for j in range(1 + i, self.rows):
                value = triangle_matrix.matrix[i][i]
                if value == 0:
                    for g in range(0, triangle_matrix.rows):
                        if triangle_matrix.matrix[g][i] != 0:
                            for k in range(0, triangle_matrix.cols):
                                triangle_matrix.matrix[i][k] += triangle_matrix.matrix[g][k]
                            value = triangle_matrix.matrix[i][i]
                            break
                value_2 = triangle_matrix.matrix[j][i]
                if value_2 == 0:
                    for g in range(0, triangle_matrix.rows):
                        if triangle_matrix.matrix[g][i] != 0:
                            for k in range(0, triangle_matrix.cols):
                                triangle_matrix.matrix[j][k] += triangle_matrix.matrix[g][k]
                            value_2 = triangle_matrix.matrix[j][i]
                            break
                for k in range(0, triangle_matrix.cols):
                    triangle_matrix.matrix[i][k] *= value_2
                    triangle_matrix.matrix[j][k] *= value
                    # print(triangle_matrix )
                for k in range(0, triangle_matrix.cols):
                    triangle_matrix.matrix[j][k] -= triangle_matrix.matrix[i][k]
        h = 0
        for i in range(self.cols - 1, 0, -1):
            for j in range(self.rows - 2 - h, -1, -1):
                value = triangle_matrix.matrix[i][i]
                if value == 0:
                    for g in range(0, triangle_matrix.rows):
                        if triangle_matrix.matrix[g][i] != 0:
                            for k in range(0, triangle_matrix.cols):
                                triangle_matrix.matrix[i][k] += triangle_matrix.matrix[g][k]
                            value = triangle_matrix.matrix[i][i]
                            break
                value_2 = triangle_matrix.matrix[j][i]
                if value_2 == 0:
                    for g in range(0, triangle_matrix.rows):
                        if triangle_matrix.matrix[g][i] != 0:
                            for k in range(0, triangle_matrix.cols):
                                triangle_matrix.matrix[j][k] += triangle_matrix.matrix[g][k]
                            value_2 = triangle_matrix.matrix[j][i]
                            break
                for k in range(0, triangle_matrix.cols):
                    triangle_matrix.matrix[i][k] *= value_2
                    triangle_matrix.matrix[j][k] *= value
                for k in range(0, triangle_matrix.cols):
                    triangle_matrix.matrix[j][k] -= triangle_matrix.matrix[i][k]
            h += 1
        for i in range(0, self.rows):
            value = triangle_matrix.matrix[i][i]
            for k in range(0, triangle_matrix.cols):
                triangle_matrix.matrix[i][k] /= value

        triangle_matrix.matrix = triangle_matrix.matrix[:, self.cols:triangle_matrix.cols]
        triangle_matrix.cols = self.cols
        return triangle_matrix

