import numpy as np
import pytest

from omo.pagerank.methods import pagerank_eigenvector

def test_pagerank():
    probability_matrix = np.array([[0,0.5,0,0.5],
                                   [0,0,1,0],
                                   [0.33,0.33,0,0.33],
                                   [0,0,1,0]])
    sth = pagerank_eigenvector(probability_matrix)
    result = np.array([0.14,0.21,0.43,0.21])
    
    assert (pytest.approx(sth) == result)