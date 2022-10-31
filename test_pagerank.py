import numpy as np
import pytest

from omo.pagerank.np_digraph import np_digraph




def test_pagerank():
    probability_matrix = np.array([[0,0.5,0,0.5],
                                   [0,0,1,0],
                                   [0.33,0.33,0,0.33],
                                   [0,0,1,0]])
    result = np.array([0.14,0.21,0.43,0.21])
    sth = np_digraph(matrix=probability_matrix)
    sth_2 = sth.pagerank(dampling_factor = 1)
    assert (pytest.approx(sth_2) == result)