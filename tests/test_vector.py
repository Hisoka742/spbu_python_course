import pytest
import numpy as np
from math import isclose
from project.vector_matrix_operations.VECTOR import Vector


def test_vector_initialization():
    vec = Vector([1, 2, 3])
    assert np.array_equal(vec, np.array([1, 2, 3])), "Vector initialization failed"


def test_vector_length():
    vec = Vector([1, 2, 3])
    assert len(vec) == 3, "Vector length is incorrect"


def test_vector_getitem():
    vec = Vector([1, 2, 3])
    assert vec[0] == 1, "Vector __getitem__ is incorrect"
    assert vec[1] == 2, "Vector __getitem__ is incorrect"
    assert vec[2] == 3, "Vector __getitem__ is incorrect"


def test_vector_dot_product():
    vec1 = Vector([1, 2, 3])
    vec2 = Vector([4, 5, 6])
    assert vec1 * vec2 == 32, "Dot product is incorrect"


def test_vector_dot_product_length_mismatch():
    vec1 = Vector([1, 2, 3])
    vec2 = Vector([4, 5])
    with pytest.raises(ValueError, match="Vectors must have the same dimensions."):
        vec1 * vec2


def test_vector_norm():
    vec = Vector([3, 4])
    assert isclose(vec.norm(), 5.0), "Vector norm calculation is incorrect"


def test_vector_angle():
    vec1 = Vector([1, 0])
    vec2 = Vector([0, 1])
    angle = vec1 ^ vec2
    assert isclose(angle, np.pi / 2), "Angle between vectors is incorrect"


def test_zero_vector_angle():
    vec1 = Vector([0, 0, 0])
    vec2 = Vector([1, 0, 0])
    with pytest.raises(
        ZeroDivisionError, match="Cannot calculate angle with zero-magnitude vector."
    ):
        vec1 ^ vec2
