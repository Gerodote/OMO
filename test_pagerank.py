import numpy as np
import pytest

from omo.pagerank.np_digraph import np_digraph, IsStochastic, find_largest_eigenvector_power_method_upto_accuracy


@pytest.mark.parametrize('matrix, result', [
    (np.array([[5,1,0],[0,25,1],[0,0,125]]), np.array([1/12000,1/100,1])), 
    (np.array([[0.  , 0.  , 0.33, 0.  ],
       [0.5 , 0.  , 0.33, 0.  ],
       [0.  , 1.  , 0.  , 1.  ],
       [0.5 , 0.  , 0.33, 0.  ]]), np.array([0.26,0.39,0.79,0.39]))
    
    ])
def test_power_method_upto_accuracy(matrix, result):
    sth2 = find_largest_eigenvector_power_method_upto_accuracy(matrix=matrix)
    assert pytest.approx(result,abs=0.01) == sth2


@pytest.mark.parametrize('matrix, result, dampling_factor, is_stochastic, left_or_right_stochastic, check_it')
def test_pagerank():
    probability_matrix = np.array([[0,0.5,0,0.5],
                                   [0,0,1,0],
                                   [0.33,0.33,0,0.33],
                                   [0,0,1,0]])
    result = np.array([0.14,0.21,0.43,0.21])
    sth = np_digraph(matrix=probability_matrix)
    sth_2 = sth.pagerank(dampling_factor=1, is_already_stochastic=IsStochastic.RIGHT)
    assert pytest.approx(result, abs=0.01) == sth_2