from typing import Optional
import numpy as np

def power_method(transition_probability_matrix:np.ndarray, iterations_qty:Optional[int]=100, dampling_factor:float = 0.85):
    '''
        https://en.wikipedia.org/wiki/Power_iteration
    '''
    eigenvector_approximation = np.random.rand(transition_probability_matrix.shape[1])
        