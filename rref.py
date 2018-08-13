"""
Reduced row echelon form

A matrix is in reduced row echelon form(also called row canonical form) if it satisfies the 
following conditions:
 * It is in row echelon form
 * Every leading coefficient is 1 and is the only nonzero entry in its column

The reduced row echelon of a matrix may be computed by Gauss-Jordan elimination.
This is an example of a matrix in reduced row echelon form, which shows that the left of the
matix is not always an identity matrix.

        | 1 0 a1 0 b1 |
    M = | 0 1 a2 0 b2 |
        | 0 0 0  0 b3 |
"""

import numpy as np


def rref(matrix):
    """transform a matrix into reduced row echelon form

    Args:
        the origial matrix
        numpy array of shape (m, n)

    Returns:
        the reduced row echelon form matrix
        numpy array of shape (m, n)

    Example:
        input: [[1, 5, 7], [-2, -7, -5]]
        >>> rref(input)
        [[1, 0, -8], [0, 1, 3]]
    """
    m, n = matrix.shape

    h, w = 0, 0 # initialization of pivot row and column

    while h < m and w < n:

        # If the entry is zero, then move to the next column
        if matrix[h][w] == 0:
            w = w + 1
        else:

            # set the entries below the pivot column to zero
            coefficient = matrix[h+1:, w] / matrix[h, w]
            element_matrix = np.identity(m - h)
            element_matrix[1:, 0] = -coefficient
            matrix[h:, :] = np.matmul(element_matrix, matrix[h:, :])
            matrix[h, :] = matrix[h, :] / matrix[h, w]

            if h > 0:
                # set the entries above the pivot column to zero
                coefficient = matrix[:h, w] / matrix[h, w]
                element_matrix = np.identity(h+1)
                element_matrix[:-1, -1] = -coefficient
                matrix[:h+1, :] = np.matmul(element_matrix, matrix[:h+1, :])

            # increase pivot row and column
            w = w + 1
            h = h + 1

    return matrix


if __name__ == "__main__":

    matrix = np.array([[1, 5, 7], [-2, -7, -5]])
    print(rref(matrix))