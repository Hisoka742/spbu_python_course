import pytest
import numpy as np
from math import isclose
from project.vector_matrix_operations.VECTOR import Vector


# Tests for Vector class functionality
class TestVector:
    def test_initialization(self):
        vector = Vector([7, 8, 9])
        assert np.array_equal(
            vector, np.array([7, 8, 9])
        ), "Initialization of vector failed"

    def test_length(self):
        vector = Vector([7, 8, 9])
        assert len(vector) == 3, "Expected vector length to be 3"

    def test_get_item(self):
        vector = Vector([7, 8, 9])
        assert vector[0] == 7, "Expected first element to be 7"
        assert vector[1] == 8, "Expected second element to be 8"
        assert vector[2] == 9, "Expected third element to be 9"

    def test_dot_product(self):
        vec_a = [7, 8, 9]
        vec_b = [10, 11, 12]
        vector_a = Vector(vec_a)
        vector_b = Vector(vec_b)
        assert vector_a * vector_b == 266, "Dot product calculation is incorrect"

    def test_dot_product_length_mismatch(self):
        vec_a = Vector([7, 8, 9])
        vec_b = Vector([10, 11])
        with pytest.raises(ValueError, match="Vectors must be of the same length"):
            vec_a * vec_b

    def test_norm_calculation(self):
        vector = Vector([6, 8])
        assert isclose(vector.norm(), 10.0), "Norm calculation is incorrect"

    def test_angle_between_vectors(self):
        vector_a = Vector([5, 0])
        vector_b = Vector([0, 5])
        angle = vector_a ^ vector_b
        assert isclose(
            angle, np.pi / 2
        ), "Calculated angle between vectors is incorrect"

    def test_zero_vector_angle_calculation(self):
        zero_vector = Vector([0, 0, 0])
        non_zero_vector = Vector([1, 2, 3])
        with pytest.raises(
            ZeroDivisionError, match="None of the vectors must have a zero magnitude"
        ):
            zero_vector ^ non_zero_vector
