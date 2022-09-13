from math import sqrt

import pytest
import sympy as sp

from src.methods import HalvingRootFinder, NewtonRootFinder
from src.general_part import NoRootOnTheRange
from src.simple_real_range import SimpleRealRange


def test_HalvingRootFinder_1():
    with pytest.raises(NoRootOnTheRange):
        var = HalvingRootFinder()(lambda x: sqrt(x) - 4,
                                0.00001,
                                SimpleRealRange(left_bound=0,
                                right_bound=15))
    
def test_HalvingRootFinder_2():
    var = HalvingRootFinder()(lambda x: sqrt(x) - 4, 0.0001, SimpleRealRange(left_bound=0, right_bound=16))
    assert(var == 16)

def test_HalvingRootFinder_3():
    var = HalvingRootFinder()(lambda x: sqrt(x+16) - 4, 0.0001, SimpleRealRange(left_bound=0, right_bound=16))
    assert(var == 0)
    
def test_HalvingRootFinder_4():
    eps = 0.0001
    var = HalvingRootFinder()(lambda x: x*x -4 , eps,SimpleRealRange(left_bound=0, right_bound=16))
    assert( abs(var - 2) < eps)


def test_NewtonRootFinder_point():
    eps = 0.0001
    x = sp.Symbol('x')
    expr = x*x - 4
    derivative_expr = sp.diff(expr)
    func_callable = sp.lambdify(x, expr, 'math')
    derivative_callable = sp.lambdify(x, derivative_expr, 'math')
    var = NewtonRootFinder()(func_callable,eps, derivative_callable, 5)
    assert( abs(var - 2) < eps )


def test_NewtonRootFinder_range():
    eps = 0.0001
    x = sp.Symbol('x')
    expr = x*x - 4
    derivative_expr = sp.diff(expr)
    func_callable = sp.lambdify(x, expr, 'math')
    derivative_callable = sp.lambdify(x, derivative_expr, 'math')
    var = NewtonRootFinder()(func_callable,eps, derivative_callable, SimpleRealRange(left_bound=0, right_bound=16))
    assert( abs(var - 2)  < eps )