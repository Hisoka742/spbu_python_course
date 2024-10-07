import numpy as np
from math import acos, sqrt
from typing import Iterable

class Vector:
    def __init__(self, data: Iterable[float | int]):
        self.vector = np.array(data)

    def __array__(self):
        return self.vector

    def __getitem__(self, index: int) -> float:
        return self.vector[index]

    def __len__(self) -> int:
        return len(self.vector)

    def __mul__(self, other: "Vector") -> float:
        if len(self.vector) != len(other.vector):
            raise ValueError("Vectors must have the same dimensions.")
        return sum(x * y for x, y in zip(self.vector, other.vector))

    def __xor__(self, other: "Vector") -> float:
        if len(self.vector) != len(other.vector):
            raise ValueError("Vectors must be of the same size.")
        if self.norm() == 0 or other.norm() == 0:
            raise ZeroDivisionError("Cannot calculate angle with zero-magnitude vector.")
        return acos((self * other) / (self.norm() * other.norm()))

    def norm(self) -> float:
        return sqrt(self * self)
