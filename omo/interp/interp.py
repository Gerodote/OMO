from enum import Enum
from typing import Callable

import numpy as np
from numpy.polynomial import Polynomial as P
import numpy.typing as npt

from ..linear_eq_solver.methods import matrix_system_equation_solver

class InterpAlgorithms(str, Enum):
    lagrange_general = "lagrange_general"
    newton_general = "newton_general"


def interp(input_x: npt.ArrayLike, input_y: npt.ArrayLike,
           type: InterpAlgorithms):

    def _lagrange_general(input_x, input_y):
        list_of_x_polynomials = []
        for j in range(len(input_x)):
            sth = P(coef=[1])
            for m in range(len(input_x)):
                if j != m:
                    denominator = input_x[j] - input_x[m]
                    sth *= P([-input_x[m] / denominator, 1 / denominator])
            list_of_x_polynomials.append(sth)
        result = P(coef=[0])
        for i in range(len(input_y)):
            result += input_y[i] * list_of_x_polynomials[i]
        return result

    def _newton_general(input_x, input_y):

        def _divided_difference_recursive_no_memoisation(input_x, input_y):
            if len(input_x) == 2:
                return (input_y[1] - input_y[0]) / (input_x[1] - input_x[0])
            else:
                return (_divided_difference_recursive_no_memoisation(
                    input_x[1:], input_y[1:]) -
                        _divided_difference_recursive_no_memoisation(
                            input_x[:-1], input_y[:-1])) / (input_x[-1] -
                                                            input_x[0])

        def _divided_difference_iterative_no_memoisation(
                x: npt.ArrayLike, y: npt.ArrayLike):
            result = 0
            for j in range(len(x)):
                sth = 1
                for i in range(len(x)):
                    if i != j:
                        sth *= x[j] - x[i]
                result += input_y[j] / sth
            return result

        result = P([input_y[0]])
        previous_poly = P([1])
        for i in range(1, len(input_x)):
            previous_poly *= P([-input_x[i - 1], 1])
            result += previous_poly * _divided_difference_iterative_no_memoisation(
                input_x[:i + 1], input_y[:i + 1])

        return result

    if type == InterpAlgorithms.lagrange_general:
        return _lagrange_general(input_x, input_y)
    elif type == InterpAlgorithms.newton_general:
        return _newton_general(input_x, input_y)


def cubic_splain(input_x, input_y, solver_of_system_of_linear_equation:Callable=matrix_system_equation_solver, result_is_piecewise=False):
    #  https://pythonnumericalmethods.berkeley.edu/notebooks/chapter17.03-Cubic-Spline-Interpolation.html
    
    n = len(input_x)
    # A*x = b, a -- matrix, b -- vector, x -- vector 
    system_of_linear_equation = np.zeros(shape=(4*(n-1),4*(n-1)),dtype=np.float64)
    vector_b = np.zeros(shape=(4*(n-1),),dtype=np.float64)
    equation_num = 0
    for j in range(n-1):
        for i in range(4):
            system_of_linear_equation[equation_num,4*j + i] = input_x[j]**(3-i)
        vector_b[equation_num] = input_y[j]
        equation_num += 1
        
    for j in range(n-1):
        for i in range(4):
            system_of_linear_equation[equation_num,4*j+i] = input_x[j+1]**(3-i)
        vector_b[equation_num] = input_y[j+1]
        equation_num += 1
    
    for j in range(n-2):
        for i in range(3):
            system_of_linear_equation[equation_num,4*j + i] = (3-i)*input_x[j+1]**(2-i)
            system_of_linear_equation[equation_num,4*(j+1) + i] = (-1)*(3-i)*input_x[j+1]**(2-i)
        vector_b[equation_num] = 0
        equation_num += 1

    for j in range(n-2):
        for i in range(2):
            system_of_linear_equation[equation_num,4*j+i] = 2*(3-(2*i))*input_x[j+1]**(1-i)
            system_of_linear_equation[equation_num,4*(j+1)+i] = (-1)*2*(3-(2*i))*input_x[j+1]**(1-i)
        vector_b[equation_num] = 0
        equation_num += 1    
   
    system_of_linear_equation[equation_num,0] = 6*input_x[0]
    system_of_linear_equation[equation_num,1] = 2
    vector_b[equation_num] = 0
    equation_num += 1
    
    system_of_linear_equation[equation_num,4*(n-2)] = 6*input_x[n-1]
    system_of_linear_equation[equation_num,4*(n-2)+1] = 2
    vector_b[equation_num] = 0
    equation_num += 1
   
    result = solver_of_system_of_linear_equation(system_of_linear_equation, vector_b)
    
    list_of_cubic_polynomials:list(P) = []
    for j in range(n-1):
        list_of_cubic_polynomials.append(P([result[4*j + 3],result[4*j+2], result[4*j + 1], result[4*j]]))
    if not result_is_piecewise:         
        return list_of_cubic_polynomials
    else:
        return lambda x: np.piecewise(
        x=x,
        condlist=[(x < input_x[1]),
                  *[((x >= input_x[i]) & (x < input_x[i + 1]))
                    for i in range(1,
                                   len(input_x) - 2)], (x >= input_x[-2])],
        funclist=list_of_cubic_polynomials)
    