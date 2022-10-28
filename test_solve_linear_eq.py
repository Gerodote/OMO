from scipy.linalg import hilbert
import numpy as np
import pytest

from omo.linear_eq_solver.methods import GEPP, jacobi_method, seidel_method


def generate_vector_b(matrix, result):
    number = matrix.shape[0]
    x = result
    to_calculate = 0
    b = np.empty(number)
    element_b = []
    while to_calculate < number:
        element_b = x * (matrix[to_calculate])
        sum_element_b = np.sum(element_b)
        b[to_calculate] = sum_element_b
        to_calculate += 1
    return b


def transform_to_diagonal_dominant_matrix(matrix):
    number_to_calculate = 0
    min_size_of_side = min(*(matrix.shape))
    while min_size_of_side > number_to_calculate:
        matrix[number_to_calculate, number_to_calculate] = 0
        matrix[matrix == matrix[number_to_calculate,
                                number_to_calculate]] = 1.1 * np.sum(
                                    np.absolute(matrix[number_to_calculate]))
        number_to_calculate += 1

    return matrix


def test_gauss_hilbert_matrix():
    matrix = hilbert(4)
    result = np.arange(4)
    vector_b = (generate_vector_b(matrix, result=result)).reshape(-1, 1)
    sth = GEPP(matrix, vector_b, pivoting=False)
    assert (pytest.approx(sth) == result)


@pytest.mark.parametrize(
    "method, matrix_arg, result_arg, eps",
    [(jacobi_method, transform_to_diagonal_dominant_matrix(
        hilbert(4)), np.arange(4), 0.01),
     (jacobi_method, transform_to_diagonal_dominant_matrix(np.random.rand(
         4, 4)), np.arange(4), 0.01),
     (seidel_method, transform_to_diagonal_dominant_matrix(
         hilbert(4)), np.arange(4), 0.01),
     (seidel_method, transform_to_diagonal_dominant_matrix(np.random.rand(
         4, 4)), np.arange(4), 0.01)])
def test_jacobi_method(method, matrix_arg, result_arg, eps):
    b = generate_vector_b(matrix=matrix_arg, result=result_arg)
    sth = method(matrix=matrix_arg,
                 vector_b=b,
                 number_of_iterations=50,
                 guess=np.ones(4))
    assert (pytest.approx(sth, abs=eps) == result_arg)
