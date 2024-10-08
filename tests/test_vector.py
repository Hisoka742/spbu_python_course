import pytest
import numpy as np
from math import isclose
from project.vector_matrix_operations.VECTOR import Vector


def test_vector_initialization():
    v = Vector([1, 2, 3])
    assert np.array_equal(v, np.array([1, 2, 3])), "Vector initialization failed"


def test_vector_length():
    v = Vector([1, 2, 3])
    assert len(v) == 3, "Vector length is incorrect"


def test_vector_getitem():
    v = Vector([1, 2, 3])
    assert v[0] == 1, "Vector __getitem__ is incorrect"
    assert v[1] == 2, "Vector __getitem__ is incorrect"
    assert v[2] == 3, "Vector __getitem__ is incorrect"


def test_vector_dot_product():
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    v1 = Vector(v1)
    v2 = Vector(v2)
    assert v2 * v1 == 32, "Dot product is incorrect"


def test_vector_dot_product_length_mismatch():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5])
    with pytest.raises(ValueError):
        v1 * v2


def test_vector_norm():
    v = Vector([3, 4])
    assert isclose(v.norm(), 5.0), "Vector norm calculation is incorrect"


def test_vector_angle():
    v1 = Vector([1, 0])
    v2 = Vector([0, 1])
    angle = v1 ^ v2
    assert isclose(angle, np.pi / 2), "Angle between vectors is incorrect"


def test_zero_vector_angle():
    v1 = Vector([0, 0, 0])
    v2 = Vector([1, 0, 0])
    with pytest.raises(ZeroDivisionError):
        v1 ^ v2
