import numpy as np
from math import acos, sqrt
from typing import Iterable

class Vector:
    def __init__(self, values: Iterable[float | int]):
        self.vector = np.array(values)

    def __array__(self):
        return self.vector

    def __getitem__(self, index: int) -> float:
        return self.vector[index]

    def __len__(self) -> int:
        return len(self.vector)

    def __mul__(self, other: "Vector") -> float:
        if len(self.vector) != len(other.vector):
            raise ValueError("Vectors must be of equal length")
        return sum(x * y for x, y in zip(self.vector, other.vector))

    def __xor__(self, other: "Vector") -> float:
        if len(self.vector) != len(other.vector):
            raise ValueError("Vectors must be of equal length")
        if self.norm() * other.norm() == 0:
            raise ZeroDivisionError("Vectors must not have zero magnitude")
        return acos((self * other) / (self.norm() * other.norm()))

    def norm(self) -> float:
        return sqrt(self * self)

class Matrix:
    def __init__(self, values: Iterable[Iterable[float | int]]):
        self.matrix = np.array(values)

    def __array__(self):
        return self.matrix

    def __repr__(self) -> str:
        return f"Matrix({self.matrix.tolist()})"

    def __getitem__(self, indices: tuple) -> float:
        return self.matrix[indices]

    def __add__(self, other: "Matrix") -> "Matrix":
        if self.matrix.shape != other.matrix.shape:
            raise ValueError("Matrices must be of the same shape")
        return Matrix(self.matrix + other.matrix)

    def __matmul__(self, other: "Matrix") -> "Matrix":
        if self.matrix.shape[1] != other.matrix.shape[0]:
            raise ValueError("Matrices have incompatible shapes")
        return Matrix(self.matrix @ other.matrix)

    def T(self) -> "Matrix":
        return Matrix(self.matrix.T)
