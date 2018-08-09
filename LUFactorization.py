"""
Implement a mathematic matrix factorization that in linear algebra and numerical analysis
For Example:
    Given a square matrix A
        | a11    a12    a13 |       | 1     0     0 |   | U11    U12    U13 |
    A = | a21    a22    a23 |  =    | L21   1     0 | * | 0      U22    U23 |
        | a31    a23    a33 |       | L31   L32   1 |   | 0      0      U33 |

    and the L is the lower triangular and U is the upper triangular matrix
"""

import numpy as np


def lu_decompose(matrix):
    """perform A = LU decomposition

    Args:
        matrix: numpy array of shape (n, n)
                a square matrix

    Returns:
        L : numpy array of shape (n, n)
            the lower triangular matrix
        U : numpy array of shape (n ,n)
            the upper triangular matrix
    """
    n = matrix.shape[0]

    # Initialization for L and U matrix
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # produce the k-th row of U
        U[i] = matrix[i] - np.matmul(L[i], U)
        # produce the k-th column of L
        L[:, i] = (matrix[:, i] - np.matmul(L, U[:, i])) / U[i, i]

    return L, U



if __name__ == "__main__":

    matrix = np.array([[2, 1, 1, 0], [4, 3, 3, 1], [8, 7, 9, 5], [6, 7, 9, 8]], dtype='float32')
    l, u = lu_decompose(matrix)
    print("L:->", l)
    print("U:->", u)