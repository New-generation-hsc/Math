"""
This file define the method to solve the determinant of a square matrix

# Determinant

In linear algebra, the determinant is a value that can be computed from the elements of square matrix.
The determinant of a matrix A is denoted det(A) or |A|

In the case of a 2x2 matrix the determinant may be defined as:
                | a   b |
        |A| =   | c   d |  = ad - bc


In the case of a 3x3 matrix, the determinant can be defined as:

        | a11  a12  a13 |       | a11  0   0 |    | 0   a12   0 |    |  0    0  a13 |
|A| =   | a21  a22  a23 |  =    | 0  a22 a23 | +  | a21  0  a23 | +  | a21  a22  0  |
        | a31  a32  a33 |       | 0  a32 a33 |    | a31  0  a33 |    | a31  a32  0  |

    = a11(a22a33 - a23a32) + a12(-a21a33 + a23a31) + a13(a21a32 - a22a31)
"""
import numpy as np


def det(square_matrix):
    """ calc the determinant of a square matrix

    Args:
        `square_matrix` is the numpy array of 2 dimensions, shape (n, n)

    Returns:
        out: a scalar value
    """

    m, n = square_matrix.shape
    assert m == n, "The input Matrix must be square"

    # the base case, where n equals one, then directly return the element
    if n == 1:
        return square_matrix[0][0]

    out = 0
    for i in range(n):
        # erase the first row and the ith column to get the cofactor
        factor = np.hstack((square_matrix[1:, :i], square_matrix[1:, i+1:]))
        out += square_matrix[0][i] * det(factor) * ((-1) ** i)

    return out


if __name__ == "__main__":

    matrix = np.array([[1]])

    print(det(matrix))

    matrix = np.array([[1, 3], [2, 4]])

    print(det(matrix))

    matrix = np.array([[0.0199626, 0.320843, 0.11461], [0.330681, 0.60768, 0.89497], [0.403389, 0.0243012, 0.48202]])

    print(det(matrix))

