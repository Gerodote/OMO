import numpy as np
import numpy.typing as npt
from typing import Optional


def find_largest_eigenvector_power_method_upto_accuracy(matrix:np.ndarray, eps=0.0001):
    '''https://en.wikipedia.org/wiki/Power_iteration'''
    b_k = np.random.rand(matrix.shape[1])
    b_k1 = b_k + np.ones(shape=b_k.shape)
    while np.norm(ord='nuc',x=b_k1-b_k) > eps:
        b_k1 = matrix @ b_k
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm

    return b_k

class np_digraph:
    def _set_value_at_diagonal(self, value):
        for i in range(self._adjacency_matrix.shape[0]):
            # print(f"Before: {self._adjacency_matrix[i,i]}")
            self._adjacency_matrix[i,i] = value
            # print(f"After: {self._adjacency_matrix[i,i]}")

    def __init__(self, matrix:Optional[np.ndarray]=None, init_size=2, dtype=np.float64):
        if matrix is not None:
            self._adjacency_matrix = matrix
            self._dtype = matrix.dtype
        else:
            self._dtype = dtype        
            self._adjacency_matrix = np.matrix(np.zeros((init_size,init_size)),dtype=self._dtype)
            self._set_value_at_diagonal(np.float64(1))

    def __str__(self) -> str:
        return str(self._adjacency_matrix.__str__()) + str(self._dtype)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def add_data_from_an_edge(self, edge_data):
        max_size_for_axis = max(edge_data['source'],edge_data['target'])
        if max_size_for_axis > self._adjacency_matrix.shape[0]:
            self._increase_size(max_size_for_axis)
            self._set_value_at_diagonal(np.float64(1))
        self._adjacency_matrix[edge_data['source'],edge_data['target']] = edge_data['weight']

    def set_data_from_ready_adjacency_matrix_numpy(self, matrix:np.ndarray):
        if matrix.ndim != 2:
            raise TypeError("matrix is not 2-dimensional. It should be [[],[],[]...]")
        if matrix.shape[0] != matrix.shape[1]:
            raise TypeError("matrix is not square.")
        self._adjacency_matrix = matrix
        self._dtype = matrix.dtype

    def get_pathes_costs(self,path):
        result = np.empty(shape=(len(path)-1,))
        for index in range(len(path)-1):
            result[index] = self._adjacency_matrix[path[index],path[index+1]]
        return result
    
    def get_amount_of_nodes(self):
        return self._adjacency_matrix.shape[0]
    
    def add_data_from_edges(self, edges_data):
        for edge in edges_data:
            self.add_data_from_an_edge(edge)
        
    def _increase_size(self, size):
        delta_size = size - self._adjacency_matrix.shape[0]
        self._adjacency_matrix = np.concatenate(
            (np.concatenate((self._adjacency_matrix, 
                        np.zeros((self._adjacency_matrix.shape[0],delta_size),dtype=self._dtype)),axis=1), 
             np.zeros((delta_size, self._adjacency_matrix.shape[1]+delta_size), dtype=self._dtype)),axis=0)

    def resize(self, size):
        if size > self._adjacency_matrix.shape[0]:
            self._increase_size(size)
    

    def set_value_single(self, to, value) -> None:
        self._adjacency_matrix[to[0],to[1]] = value 

    def set_value_dict(self, to_value) -> None:
        for item in to_value:
            self._adjacency_matrix[item[0][0], item[0][1]] = item[1]
        
    def set_value_numpy_array(self, arr:npt.NDArray):
        for i in arr:
            self._adjacency_matrix[arr[i][0],arr[i][1]] = arr[i][2]
    
    def set_value(self,
                  to,
                  value_or_values, 
                  to_value=None):
        if to_value != None:
            self.set_value_dict(to_value)
            return
        try:
            iterator = iter(to[0])
        except TypeError:
            return self.set_value_single(to, value_or_values)
        else:
            for i in range(len(to)):
                self.set_value_single(to[i],value_or_values[i])
                
                
    def out_edges_qty(self, node:int):
        if node - 1 > self._adjacency_matrix.shape[0] or node < 0: 
           raise ValueError("The node doesn't exist.")
        return np.sum(self._adjacency_matrix[node])
        
        
    def make_adjacency_matrix_right_stochastic(self):
        '''https://en.wikipedia.org/wiki/Stochastic_matrix'''
        new_matrix = np.zeros(shape=self._adjacency_matrix.shape, dtype=np.float64)
        for j in self._adjacency_matrix:
            out_edges_qty = self.out_edges_qty(j)
            if out_edges_qty != 0:
                new_matrix[j] = self._adjacency_matrix[j] / out_edges_qty
            else:
                new_matrix[j] = np.ones(shape=new_matrix.shape[j],dtype=np.float64) / out_edges_qty
        self._adjacency_matrix = new_matrix


    def make_adjacency_matrix_left_stochastic(self):
        '''https://en.wikipedia.org/wiki/Stochastic_matrix'''
        new_matrix = np.zeros(shape=self._adjacency_matrix.shape, dtype=np.float64)
        for j in self._adjacency_matrix[:,]:
            out_edges_qty = self.out_edges_qty(j)
            if out_edges_qty != 0:
                new_matrix[:,j] = self._adjacency_matrix[j] / out_edges_qty
            else:
                new_matrix[:,j] = np.ones(shape=new_matrix.shape[j],dtype=np.float64) / out_edges_qty
        self._adjacency_matrix = new_matrix
    

    
    def pagerank(self, personal_vector:Optional[np.ndarray]=None, dampling_factor:float=0.82, eps=0.0001, optimization_eps:bool=True):
        '''
        power iteration method used for finding the largest eigenvector approximation
        
        https://ipsen.math.ncsu.edu/ps/slides_imacs.pdf
        https://www.rose-hulman.edu/~bryan/googleFinalVersionFixed.pdf
        https://en.wikipedia.org/wiki/PageRank
        '''
        
        self.make_adjacency_matrix_left_stochastic()
        vector = None
        if personal_vector is not None:
            if vector.shape[0] != 1 and vector.shape[1] != self._adjacency_matrix.shape[0]:
                raise TypeError(f"The vector has not appropriate size. It has {vector.shape[0]} on {vector.shape[1]}. It should be 1 on {self._adjacency_matrix.shape[0]}")
            vector = personal_vector / np.norm(x=personal_vector, ord='nuc')
        else:
            vector = np.ones(shape=(self._adjacency_matrix.shape[0],1), dtype=np.float64)
            
        
        matrix_G =(dampling_factor*self._adjacency_matrix) + ((1 - dampling_factor)*np.eye(self._adjacency_matrix.shape[0]) @ personal_vector)
        
        our_eps = eps
        if optimization_eps == True:
            our_eps = eps * matrix_G.shape[0]
        
        return find_largest_eigenvector_power_method_upto_accuracy(matrix_G, our_eps)
        
        
        