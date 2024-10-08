import pytest
import numpy as np
from math import isclose
from project.vector_matrix_operations.VECTOR import Vector


def test_initialize_vector():
    """Test initialization of a vector with three elements."""
    vec = Vector([4, 5, 6])
    expected = np.array([4, 5, 6])
    assert np.array_equal(vec, expected), "Vector did not initialize correctly."


def test_vector_size():
    """Test vector length calculation."""
    vec = Vector([7, 8, 9, 10])
    assert len(vec) == 4, "Vector length should be 4, but it is incorrect."


def test_access_vector_elements():
    """Test accessing individual elements in the vector."""
    vec = Vector([3, 1, 4, 1, 5])
    assert vec[0] == 3, "Element at index 0 should be 3."
    assert vec[2] == 4, "Element at index 2 should be 4."
    assert vec[4] == 5, "Element at index 4 should be 5."


def test_vector_inner_product():
    """Test the dot product of two vectors."""
    vec1 = Vector([2, 4, 6])
    vec2 = Vector([1, 3, 5])
    result = vec1 * vec2
    expected_dot_product = 44
    assert (
        result == expected_dot_product
    ), f"Expected dot product to be {expected_dot_product}, but got {result}."


def test_dot_product_length_mismatch():
    """Test for length mismatch during dot product calculation."""
    vec1 = Vector([5, 6, 7])
    vec2 = Vector([3, 2])
    with pytest.raises(ValueError, match="Vectors must be of the same length"):
        vec1 * vec2


def test_vector_magnitude():
    """Test the Euclidean norm (magnitude) of a vector."""
    vec = Vector([6, 8])
    expected_norm = 10.0
    assert isclose(
        vec.norm(), expected_norm
    ), f"Vector norm should be {expected_norm}, but is incorrect."


def test_vectors_angle():
    """Test angle between two perpendicular vectors."""
    vec1 = Vector([0, 1])
    vec2 = Vector([1, 0])
    angle = vec1 ^ vec2
    expected_angle = np.pi / 2  # 90 degrees in radians
    assert isclose(
        angle, expected_angle
    ), f"Expected angle to be {expected_angle} radians, but got {angle}."


def test_angle_with_zero_vector():
    """Test that angle calculation with a zero vector raises an error."""
    vec1 = Vector([0, 0, 0])
    vec2 = Vector([2, 0, 0])
    with pytest.raises(
        ZeroDivisionError, match="None of the vectors must have a zero magnitude"
    ):
        vec1 ^ vec2
