from math import sqrt

import pytest
import sympy as sp

from omo.root_finder.methods import HalvingRootFinder, NewtonRootFinder
from omo.root_finder.general_part import NoRootOnTheRange
from omo.root_finder.simple_real_range import SimpleRealRange


@pytest.mark.parametrize("foo, eps, left, right",
                         [(lambda x: sqrt(x) - 4, 0.00001, 0, 15),
                          (lambda x: sqrt(x) - 4, 0.0001, 17, 20)])
def test_HalvingRootFinder_exception(foo, eps, left, right):
    with pytest.raises(NoRootOnTheRange):
        var = HalvingRootFinder(func=foo,
                                eps=eps,
                                range=SimpleRealRange(left_bound=left,
                                                      right_bound=right))


@pytest.mark.parametrize("foo, eps, left, right, result",
                         [(lambda x: sqrt(x) - 4, 0.0001, 12, 20, 16),
                          (lambda x: sqrt(x + 16) - 4, 0.0001, 0, 16, 0),
                          (lambda x: x * x - 4, 0.0001, 0, 16, 2)])
def test_HalvingRootFinder_value(foo, eps, left, right, result):
    var = HalvingRootFinder(foo,
                            eps=eps,
                            range=SimpleRealRange(left_bound=left,
                                                  right_bound=right))
    assert (var == result or abs(var - result) < eps)


def test_NewtonRootFinder_point():
    eps = 0.0001
    x = sp.Symbol('x')
    expr = x * x - 4
    derivative_expr = sp.diff(expr)
    func_callable = sp.lambdify(x, expr, 'math')
    derivative_callable = sp.lambdify(x, derivative_expr, 'math')
    var = NewtonRootFinder(func_callable,
                           eps=eps,
                           derivative=derivative_callable,
                           range_or_first_point=5)
    assert (abs(var - 2) < eps)


def test_NewtonRootFinder_range():
    eps = 0.0001
    x = sp.Symbol('x')
    expr = x * x - 4
    derivative_expr = sp.diff(expr)
    func_callable = sp.lambdify(x, expr, 'math')
    derivative_callable = sp.lambdify(x, derivative_expr, 'math')
    var = NewtonRootFinder(func_callable,
                           eps=eps,
                           derivative=derivative_callable,
                           range_or_first_point=SimpleRealRange(
                               left_bound=0, right_bound=16))
    assert (abs(var - 2) < eps)
