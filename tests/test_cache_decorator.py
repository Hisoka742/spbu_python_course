import pytest
from project.curry_uncurry_cache.curry_uncurry_cache import cache_results

# This test checks if the cache decorator works as expected. The `cache_results`
# decorator is used to cache the results of the `add` function based on its input
# arguments, with a limit on the maximum cache size.


def test_cache_decorator():
    # Apply the `cache_results` decorator to the `add` function, with a maximum
    # cache size of 2. This means only the last 2 results will be cached.
    @cache_results(max_size=2)
    def add(x, y):
        # The function simply returns the sum of x and y
        return x + y

    # First call to `add(1, 2)`. The result will be calculated and cached.
    assert add(1, 2) == 3

    # Second call to `add(1, 2)`. This result should come from the cache, as it's
    # the same input. No recalculation should happen.
    assert add(1, 2) == 3  # Should be from cache

    # Call to `add(2, 3)`. A new result is computed and cached.
    assert add(2, 3) == 5

    # Call to `add(3, 4)`. The result is computed and cached.
    # Now the cache is full (contains results for `add(1, 2)` and `add(2, 3)`).
    assert add(3, 4) == 7

    # At this point, the cache size limit (2) has been exceeded. The result of
    # `add(1, 2)` is the oldest and should have been evicted from the cache.
    # When we call `add(1, 2)` again, the result will be recalculated.
    assert add(1, 2) == 3  # Recalculated since it was evicted from the cache


def test_cache_with_value_checking():
    call_counter = [0]

    @cache_results(max_size=2)
    def add(x, y):
        call_counter[0] += 1
        return x + y

    # First call, cache is empty, function is executed
    assert add(1, 2) == 3
    assert call_counter[0] == 1  # Function was executed once

    # Second call with same arguments, should be from cache
    assert add(1, 2) == 3
    assert call_counter[0] == 1  # No new execution, from cache

    # New arguments, new execution
    assert add(2, 3) == 5
    assert call_counter[0] == 2  # Function was executed again

    # Adding a new entry, causing the oldest to be evicted
    assert add(3, 4) == 7
    assert call_counter[0] == 3  # Function executed

    # Oldest entry evicted, should execute again
    assert add(1, 2) == 3
    assert call_counter[0] == 4  # Executed again, cache evicted
