import pytest
import numpy as np
from project.vector_matrix_operations.Matrix import Matrix

# Tests for Matrix class functionality
class TestMatrix:

    def test_get_item(self):
        matrix = Matrix([[1, 2], [3, 4]])
        assert matrix[0, 1] == 2, "Expected element at (0, 1) to be 2"
        assert matrix[1, 0] == 3, "Expected element at (1, 0) to be 3"

    def test_repr_format(self):
        matrix = Matrix([[1, 2], [3, 4]])
        expected_representation = "Matrix([[1, 2], [3, 4]])"
        assert repr(matrix) == expected_representation, f"Expected repr to be {expected_representation}"

    def test_string_format(self):
        matrix = Matrix([[1, 2], [3, 4]])
        expected_string = "[[1 2]\n [3 4]]"
        assert str(matrix) == expected_string, f"Expected str to be {expected_string}"

    def test_addition(self):
        mat_a = Matrix([[1, 2], [3, 4]])
        mat_b = Matrix([[5, 6], [7, 8]])
        sum_result = mat_a + mat_b
        expected_sum = Matrix([[6, 8], [10, 12]])
        assert np.array_equal(sum_result, expected_sum), "Matrix addition result is incorrect"

    def test_add_with_zero_matrix(self):
        mat_a = Matrix([[1, 2], [3, 4]])
        zero_mat = Matrix([[0, 0], [0, 0]])
        sum_result = mat_a + zero_mat
        assert np.array_equal(sum_result, mat_a), "Adding zero matrix should return original matrix"

    def test_shape_mismatch_addition(self):
        mat_a = Matrix([[1, 2], [3, 4]])
        mat_b = Matrix([[5, 6, 7], [8, 9, 10]])
        with pytest.raises(ValueError, match="Matrices must be of the same shape"):
            mat_a + mat_b

    def test_multiplication(self):
        mat_a = Matrix([[1, 2], [3, 4]])
        mat_b = Matrix([[5, 6], [7, 8]])
        product_result = mat_a @ mat_b
        expected_product = Matrix([[19, 22], [43, 50]])
        assert np.array_equal(product_result, expected_product), "Matrix multiplication result is incorrect"

    def test_non_square_multiplication(self):
        mat_a = Matrix([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
        mat_b = Matrix([[7, 8], [9, 10], [11, 12]])  # 3x2 matrix

        product_result = mat_a @ mat_b
        expected_shape = (2, 2)
        expected_values = np.array([[58, 64], [139, 154]])

        assert product_result.matrix.shape == expected_shape
        assert np.array_equal(product_result.matrix, expected_values)

    def test_identity_multiplication(self):
        mat_a = Matrix([[1, 2], [3, 4]])
        identity_matrix = Matrix([[1, 0], [0, 1]])
        product_result = mat_a @ identity_matrix
        assert np.array_equal(product_result, mat_a), "Multiplying by identity matrix should return original matrix"

    def test_shape_mismatch_multiplication(self):
        mat_a = Matrix([[1, 2], [3, 4]])
        mat_b = Matrix([[5, 6, 7]])
        with pytest.raises(ValueError, match="Matrix shapes are not compatible for multiplication"):
            mat_a @ mat_b

    def test_transpose_functionality(self):
        mat_a = Matrix([[1, 2], [3, 4]])
        transposed_result = mat_a.T()
        expected_transpose = Matrix([[1, 3], [2, 4]])
        assert np.array_equal(transposed_result, expected_transpose), "Matrix transpose result is incorrect"
