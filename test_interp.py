from numpy.polynomial import polyutils as pu
import pytest
from matplotlib import pyplot as plt
import numpy as np

from omo.interp.interp import InterpAlgorithms, interp, cubic_splain


@pytest.mark.parametrize(
    'input_x, input_y, type, polynome_result',
    [([1, 2, 3], [1, 4, 9], InterpAlgorithms("lagrange_general"), [0, 0, 1]),
     ([1, 2, 3, 4], [1, 8, 27, 64
                     ], InterpAlgorithms("newton_general"), [0, 0, 0, 1])])
def test_interp_general(input_x, input_y, type, polynome_result):
    result = interp(input_x, input_y, type)
    assert pytest.approx(polynome_result, abs=0.01) == pu.as_series(result)


@pytest.mark.parametrize(
    'input_x, input_y, polynomes_result, check_result_for_value, its_result',
    [([0, 1, 2], [1, 3, 2], [[1, 2.75, 0, -0.75], [-0.5, 7.25, -4.5, 0.75]
                             ], 1.5, 2.7813)])
def test_interp_cubic_spline(input_x, input_y, polynomes_result,
                             check_result_for_value, its_result):
    result = cubic_splain(input_x, input_y)
    for i in range(len(input_x) - 1):
        assert pytest.approx(polynomes_result[i],
                             abs=0.01) == pu.as_series(result[i])

    x = check_result_for_value
    result_value = np.piecewise(
        x=x,
        condlist=[(x < input_x[1]),
                  *[((x >= input_x[i]) & (x < input_x[i + 1]))
                    for i in range(1,
                                   len(input_x) - 2)], (x >= input_x[-2])],
        funclist=result)
    assert pytest.approx(its_result, abs=0.01) == result_value
    
    result = cubic_splain(input_x, input_y, result_is_piecewise=True)
    assert pytest.approx(its_result, abs=0.01) == result(check_result_for_value)


if __name__ == '__main__':
    x_input = [0, 1, 2, 3, 4]
    y_input = [4, 2, -1, 3, 5]

    res_lagrange = interp(x_input,
                          y_input,
                          type=InterpAlgorithms.lagrange_general)
    res_newton = interp(x_input, y_input, type=InterpAlgorithms.newton_general)

    res_spline = cubic_splain(x_input, y_input)

    x = np.linspace(-0.5 + x_input[0], 0.5 + x_input[-1], 100)
    y_for_plot_lagrange = res_lagrange(x)
    y_for_plot_newton = res_newton(x)

    y_for_plot_cubic = np.piecewise(
        x=x,
        condlist=[(x < x_input[1]),
                  *[((x >= x_input[i]) & (x < x_input[i + 1]))
                    for i in range(1,
                                   len(x_input) - 2)], (x >= x_input[-2])],
        funclist=res_spline)

    plt.scatter(x_input, y_input, label='input', color='red', linewidth=15)

    plt.plot(x,
             y_for_plot_lagrange,
             label='lagrange',
             color='blue',
             alpha=0.5,
             linewidth=5)
    plt.plot(x,
             y_for_plot_newton,
             label='newton',
             color='pink',
             ls='--',
             linewidth=2)
    plt.plot(x,
             y_for_plot_cubic,
             label='cubic splain',
             color='green',
             alpha=0.5,
             linewidth=4,
             ls='-.')
    plt.legend(loc='lower right')
    plt.show()