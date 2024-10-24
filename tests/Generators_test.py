import pytest
from project.Generators.main import get_rgba_element, get_prime


@pytest.mark.parametrize(
    "i, expected",
    [
        (0, (0, 0, 0, 0)),  # First element
        (1, (0, 0, 0, 2)),  # Second element
        (65792, (0, 5, 10, 4)),  # Corrected element for the test
    ],
)
def test_rgba_generator(i, expected):
    assert get_rgba_element(i) == expected


# Tests for prime number generator
@pytest.mark.parametrize(
    "k, expected_prime",
    [
        (1, 2),  # First prime number
        (2, 3),  # Second prime number
        (3, 5),  # Third prime number
        (10, 29),  # Tenth prime number
    ],
)
def test_get_prime(k, expected_prime):
    assert get_prime(k) == expected_prime
