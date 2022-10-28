from abc import ABC, abstractmethod
from typing import Callable


class AbstractRealRange(ABC):

    @abstractmethod
    def get_left_bound(self):
        pass

    @abstractmethod
    def get_right_bound(self):
        pass


class NoRootOnTheRange(ValueError):
    pass


class HugeRange(ValueError):
    pass