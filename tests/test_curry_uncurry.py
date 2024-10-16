import pytest
from project.curry_uncurry_cache.curry_uncurry_cache import (
    curry_explicit,  # Curries a function by taking an explicit arity
    uncurry_explicit,  # Uncurries a curried function
    InvalidArityException,  # Custom exception raised for invalid arity
)

# Test currying with basic example
def test_curry_basic():
    # Define a simple function `f` that takes three arguments and returns a string representation
    f = lambda x, y, z: f"<{x},{y},{z}>"

    # Curry `f` with an arity of 3, meaning it will take one argument at a time
    f_curried = curry_explicit(f, 3)

    # Test the curried function, making sure it works correctly by calling it with one argument at a time
    assert f_curried(1)(2)(3) == "<1,2,3>"


# Test uncurrying with basic example
def test_uncurry_basic():
    # Define the same function `f`
    f = lambda x, y, z: f"<{x},{y},{z}>"

    # Curry `f` with an arity of 3
    f_curried = curry_explicit(f, 3)

    # Uncurry the curried function, so it can now take all 3 arguments at once
    f_uncurried = uncurry_explicit(f_curried, 3)

    # Test the uncurried function by passing all arguments at once
    assert f_uncurried(1, 2, 3) == "<1,2,3>"


# Test invalid arity during currying
def test_curry_invalid_arity():
    # Define a simple function `f` that takes two arguments and returns their sum
    f = lambda x, y: x + y

    # Expect an InvalidArityException when attempting to curry with a negative arity
    with pytest.raises(InvalidArityException, match="Arity cannot be negative"):
        curry_explicit(f, -1)  # Negative arity should trigger the exception


# Test invalid arity during uncurrying
def test_uncurry_invalid_arity():
    # Define a simple function `f` that takes two arguments and returns their sum
    f = lambda x, y: x + y

    # Curry the function with an arity of 2
    f_curried = curry_explicit(f, 2)

    # Uncurry the function to allow taking both arguments at once
    f_uncurried = uncurry_explicit(f_curried, 2)

    # Expect an InvalidArityException when the uncurried function is called with the wrong number of arguments (e.g., 1 instead of 2)
    with pytest.raises(InvalidArityException, match="Expected 2 arguments, got 1"):
        f_uncurried(1)  # Only 1 argument, but expected 2
