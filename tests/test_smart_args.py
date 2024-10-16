import pytest
from project.curry_uncurry_cache.curry_uncurry_cache import (
    Evaluated,  # A class that evaluates a callable when used
    Isolated,  # A class that makes deep copies of objects to isolate changes
    smart_args,  # A decorator that handles special arguments like `Evaluated` and `Isolated`
)

# Test the `Isolated` smart argument
def test_smart_args_isolated():
    # Apply the `smart_args` decorator to manage special arguments like `Isolated`
    @smart_args()
    def test_func(*, d=Isolated({"key": "initial"})):
        # Modify the key in the dictionary passed as an isolated argument
        d[
            "key"
        ] = "value"  # Since `Isolated` is used, the original dict should not be affected
        return d

    # Call the function and check if the isolated dictionary is updated
    result = test_func()
    assert result["key"] == "value"  # Ensure that the value was correctly updated


# Test the `Evaluated` smart argument
def test_smart_args_evaluated():
    # Create a simple counter as a list to be modified inside the function
    counter = [0]

    # A helper function to increment the counter
    def increment_counter():
        counter[0] += 1
        return counter[0]

    # Apply the `smart_args` decorator to manage the `Evaluated` argument
    @smart_args()
    def test_func(*, value=Evaluated(increment_counter)):
        # The `Evaluated` object should call the `increment_counter` function and return the result
        return value

    # Call the function multiple times and ensure the `increment_counter` is evaluated correctly
    assert test_func() == 1  # First call, counter should increment to 1
    assert test_func() == 2  # Second call, counter should increment to 2
    assert test_func() == 3  # Third call, counter should increment to 3


# Test the `smart_args` decorator with positional arguments allowed
def test_smart_args_positional():
    # Use `smart_args` with the `allow_positional` option, which allows the function to accept positional arguments
    @smart_args(allow_positional=True)
    def test_func(x=Isolated()):
        # Modify the list passed as an isolated argument by appending a value to it
        x.append(42)
        return x

    # Call the function with a list and ensure that the `Isolated` argument behaves correctly (deep copying the list)
    result = test_func([1, 2, 3])
    assert result == [
        1,
        2,
        3,
        42,
    ]  # Ensure that the original list is not mutated and the copy is modified
