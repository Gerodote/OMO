from .general_part import AbstractRealRange


class SimpleRealRange(AbstractRealRange):

    def __init__(self, left_bound: float, right_bound: float):
        self.left_bound: float = left_bound
        self.right_bound: float = right_bound

    def get_left_bound(self) -> float:
        return self.left_bound

    def get_right_bound(self) -> float:
        return self.right_bound

    def __eq__(self, __o: object) -> bool:
        return (self.left_bound == __o.left_bound) and (self.right_bound
                                                        == __o.right_bound)

    def __hash__(self) -> int:
        return hash((self.left_bound, self.right_bound))
