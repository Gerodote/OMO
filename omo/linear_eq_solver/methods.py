import numpy as np


class DimensionalError(TypeError):
    ...


class SizeError(TypeError):
    ...

def matrix_system_equation_solver(A,b):
    return np.linalg.inv(A) @ b


def GEPP(A: np.ndarray, b: np.ndarray, pivoting: bool = True):
    """
    Gaussian elimination with partial pivoting.
    input: A is an n x n numpy matrix
           b is an n x 1 numpy array
    output: x is the solution of Ax=b
            with the entries permuted in
            accordance with the pivoting
            done by the algorithm
    :return
    """

    def _validate_input(A, b):
        if A.ndim != 2:
            raise DimensionalError("A matrix should have 2 dimensions")
        if b.ndim != 2:
            raise DimensionalError(
                "b should be as [[value_1],[value_2], ..., [value_n]]")
        if b.shape[0] != A.shape[1]:
            raise SizeError(
                "Invalid argument: incompatible sizes between" + "A & b.",
                b.shape[0], A.shape[1])

    def _elimination(A, b, pivoting):

        # Elimination
        for k in range(A.shape[1] - 1):
            if pivoting:
                # Pivot
                maxindex = abs(A[k:, k]).argmax() + k
                if A[maxindex, k] == 0:
                    raise ValueError("Matrix is singular.")
                # Swap
                if maxindex != k:
                    A[[k, maxindex]] = A[[maxindex, k]]
                    b[[k, maxindex]] = b[[maxindex, k]]
            else:
                if A[k, k] == 0:
                    raise ValueError(
                        "Pivot element is zero. Try setting pivoting to True.")
            # Eliminate
            for row in range(k + 1, A.shape[1]):
                multiplier = A[row, k] / A[k, k]
                A[row, k:] = A[row, k:] - multiplier * A[k, k:]
                b[row] = b[row] - multiplier * b[k]

    def _back_substitution(A, b):

        x = np.zeros(A.shape[1])
        for k in range(A.shape[1] - 1, -1, -1):
            x[k] = (b[k] - np.dot(A[k, k + 1:], x[k + 1:])) / A[k, k]
        return x

    A_copy = A.copy()
    b_copy = b.copy()

    _validate_input(A_copy, b_copy)
    _elimination(A_copy, b_copy, pivoting)
    return _back_substitution(A_copy, b_copy)


def is_matrix_positive_definite(matrix: np.ndarray):
    calculate = 0
    while min(*(matrix.shape)) > calculate:
        data_without_column = np.delete(matrix, calculate, 1)
        data_without_row = np.delete(data_without_column, calculate, 0)
        if np.linalg.det(data_without_row) != 0:
            calculate += 1
        else:
            return False
    return True


def is_diagonal_dominant_matrix(matrix) -> bool:
    rows: int = 0
    columns: int = 0
    min_size_of_side: int = min(*matrix.shape)
    for_do_while: bool = True
    while for_do_while:
        if abs(matrix[rows, columns]) > np.sum(
                np.absolute(np.delete((matrix[rows]), columns))):
            rows += 1
            columns += 1
        if min_size_of_side - 1 == rows and min_size_of_side - 1 == columns:
            return True
        else:
            for_do_while = False
        if min_size_of_side <= rows:
            for_do_while = False
    return False


def jacobi_method(matrix: np.ndarray,
                  vector_b: np.ndarray,
                  number_of_iterations: int,
                  guess: np.ndarray,
                  is_to_check: bool = False):
    '''check works badly'''
    if is_to_check:
        if is_matrix_positive_definite(matrix):
            raise ValueError("The method diverges for the matrix.")
    b = vector_b
    D = np.diag(np.diag(matrix))
    R = matrix - D
    x = guess
    for i in range(number_of_iterations):
        x = np.dot(np.linalg.inv(D), (b - np.dot(R, x)))
    return x


def seidel_method(matrix: np.ndarray,
                  vector_b: np.ndarray,
                  number_of_iterations: int,
                  guess: np.ndarray,
                  is_to_check: bool = False):
    '''check works badly'''
    if is_to_check:
        if not is_matrix_positive_definite(
                matrix) or not is_matrix_positive_definite(matrix):
            raise ValueError("The method diverges for the matrix.")
    b = vector_b
    n = number_of_iterations
    x = guess
    L = np.tril(matrix)
    U = matrix - L
    for i in range(n):
        x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
    return x
