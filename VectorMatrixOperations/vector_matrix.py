import math
from typing import List

# Операции над векторами
def dot_product(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    return sum(x * y for x, y in zip(v1, v2))

def vector_length(v: List[float]) -> float:
    return math.sqrt(dot_product(v, v))

def angle_between(v1: List[float], v2: List[float]) -> float:
    dp = dot_product(v1, v2)
    length_v1 = vector_length(v1)
    length_v2 = vector_length(v2)
    if length_v1 == 0 or length_v2 == 0:
        raise ValueError("Vectors must be non-zero length")
    return math.acos(dp / (length_v1 * length_v2))

# Операции над матрицами
def matrix_addition(m1: List[List[float]], m2: List[List[float]]) -> List[List[float]]:
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        raise ValueError("Matrices must have the same dimensions")
    return [[m1[i][j] + m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]

def matrix_multiplication(m1: List[List[float]], m2: List[List[float]]) -> List[List[float]]:
    if len(m1[0]) != len(m2):
        raise ValueError("Number of columns in the first matrix must equal number of rows in the second matrix")
    result = [[0] * len(m2[0]) for _ in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            result[i][j] = sum(m1[i][k] * m2[k][j] for k in range(len(m2)))
    return result

def matrix_transpose(matrix: List[List[float]]) -> List[List[float]]:
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

