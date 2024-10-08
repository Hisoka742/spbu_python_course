import numpy as np
from typing import Iterable


class Matrix:
    def __init__(self, data: Iterable[Iterable[float | int]]):
        self.matrix = np.array(data)

    def __array__(self):
        return self.matrix

    def __repr__(self) -> str:
        return f"Matrix({self.matrix.tolist()})"

    def __str__(self) -> str:
        return str(self.matrix)

    def __getitem__(self, indices: tuple) -> float:
        return self.matrix[indices]

    @property
    def data(self):
        return self.matrix

    def __add__(self, other: "Matrix") -> "Matrix":
        if self.matrix.shape != other.matrix.shape:
            raise ValueError("Shape mismatch: cannot add matrices")
        return Matrix(self.matrix + other.matrix)

    def __matmul__(self, other: "Matrix") -> "Matrix":
        if self.matrix.shape[1] != other.matrix.shape[0]:
            raise ValueError("Shape mismatch: cannot multiply matrices")
        return Matrix(self.matrix @ other.matrix)

    def T(self) -> "Matrix":
        return Matrix(self.matrix.T)
