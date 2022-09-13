from typing import Callable, Union
from random import random

from .general_part import NoRootOnTheRange, HugeRange, AbstractRealRange, AbstractRootFinder


class HalvingRootFinder(AbstractRootFinder):

    def __call__(self, func: Callable, eps: float,
                 range: AbstractRealRange) -> float:
        a = range.get_left_bound()
        b = range.get_right_bound()
        if func(a) == 0:
            return a
        if func(b) == 0:
            return b
        c = (a + b) / 2
        print(c)
        while b - a > eps:
            if func(c) == 0:
                break
            if func(c) * func(b) <= 0:
                a = c
            if func(a) * func(c) <= 0:
                b = c
            if func(a) * func(b) > 0:
                raise NoRootOnTheRange(
                    "Either there's no root on the range, or it exist, but you should use another method."
                )
            print(c)
            c = (a + b) / 2
        return c

class NewtonRootFinder(AbstractRootFinder):
    def __call__(self, func: Callable[[float], float], eps: float, derivative:Callable[[float], float], range_or_first_point:Union[AbstractRealRange, float]) -> float:
        start_point = None
        if isinstance(range_or_first_point, AbstractRealRange):
            start_point = random()*(range_or_first_point.get_right_bound() - range_or_first_point.get_left_bound()) + range_or_first_point.get_left_bound()
        else:
            start_point = range_or_first_point

        next_point = start_point - func(start_point)/derivative(start_point)
        while abs(next_point - start_point) >= eps:
            start_point = next_point
            next_point = start_point - func(start_point)/derivative(start_point)
        return next_point

class RelaxationRootFinder(AbstractRootFinder):
    def __call__(self, func:Callable[[float], float], eps:float, derivative:Callable[[float], float], range:AbstractRealRange):
        # You cannot do this mathod in rigth way. 
        # How would you find min/max of a derivative on a range?
        # It's an algorithm from neural networks: find min/max using gradient and do some interesting stuff to find it faster.
        # Sorry, but this method is great for theoretical purposes.
        pass
        
        