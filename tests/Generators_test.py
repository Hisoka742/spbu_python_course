import pytest
from project.Generators.main import get_nth_rgba, get_prime

@pytest.mark.parametrize("index, expected_rgba", [
    (0, (0, 0, 0, 0)),
    (1, (0, 0, 0, 2)),
    (256, (0, 0, 1, 0)),
    (65536, (0, 1, 0, 0)),
    (16777216, (1, 0, 0, 0)),
])
def test_get_nth_rgba(index, expected_rgba):
    """Test RGBA generator for specific indices."""
    assert get_nth_rgba(index) == expected_rgba


@pytest.mark.parametrize("index, expected_prime", [
    (1, 2),
    (2, 3),
    (3, 5),
    (4, 7),
    (5, 11),
    (6, 13),
    (10, 29),
])
def test_get_prime(index, expected_prime):
    """Test prime number generator for specific indices."""
    assert get_prime(index) == expected_prime
